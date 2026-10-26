import csv
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.schemas.csv_import import CsvEncoding, ImportKind

COMPANY_FIELDS = {
    "SYNTHETIC_U": "id",
    "GROUP_ID": "group_id",
    "ORG_TYPE": "org_type",
    "OKFS_NAME": "ownership_form",
    "COMPANY_AGE_YEARS_GRADATION": "age_category",
    "ECO_MARK_LABEL": "development_stage",
    "MAINOKVED_CODE": "okved_code",
    "MAINOKVED_NAME": "okved_name",
    "NEGATIVEMARK": "key_owner_flag",
    "AMOUNT_WORKERS_FNS_GRADATION": "employees_fns_category",
    "AMOUNT_WORKERS_SPARK_GRADATION": "employees_spark_category",
    "CL_TYPE": "client_type",
    "CL_PRODUCT_RKO_MARK": "has_rko",
    "CL_PRODUCT_GAR_MARK": "has_guarantees",
    "CL_PRODUCT_LOA_MARK": "has_credit",
    "CL_PRODUCT_LEA_MARK": "has_leasing",
    "CL_PRODUCT_DEP_MARK": "has_deposit",
    "CL_PRODUCT_ZPP_MARK": "has_payroll",
    "FTS_FLAG": "has_fts_data",
    "FTS_COUNTRY_NAME": "fts_country_name",
    "FACTORING_MARK": "has_factoring",
    "REVENUE_COMPANY": "annual_revenue",
    "CITY": "city",
    "REGION": "region",
}
PERSON_FIELDS = {"SYNTHETIC_F", "SYNTHETIC_U", "CITY_NAME", "RC_NAME"}
PAYMENT_FIELDS = {"PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"}
BOOLEAN_FIELDS = {name for name in COMPANY_FIELDS.values() if name.startswith("has_")} | {
    "key_owner_flag"
}
NULLS = {"", "null", "none", "nan", "na", "n/a"}


class CsvValidationError(ValueError):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("CSV validation failed")


@dataclass
class ParsedCsv:
    rows: list[dict]
    delimiter: str
    skipped_duplicate_rows: int = 0
    person_attribute_conflicts: int = 0


def optional(value: str) -> str | None:
    value = value.strip()
    return None if value.lower() in NULLS else value


def identifier(value: str, field: str) -> str:
    # IDs are opaque text: keep leading zeroes and do not treat literal 'NA' as null.
    value = value.strip()
    if not value or len(value) > 512:
        raise ValueError(f"{field}: required identifier, maximum 512 characters")
    return value


def boolean(value: str, field: str) -> bool | None:
    value = optional(value)
    if value is None:
        return None
    if value.lower() in {"1", "1.0", "true", "yes", "да"}:
        return True
    if value.lower() in {"0", "0.0", "false", "no", "нет"}:
        return False
    raise ValueError(f"{field}: expected 0/1 or true/false")


def number(value: str, field: str, *, required: bool = False) -> Decimal | None:
    value = optional(value)
    if value is None:
        if required:
            raise ValueError(f"{field}: required number")
        return None
    try:
        result = Decimal(value.replace(" ", "").replace("\u00a0", "").replace(",", "."))
    except InvalidOperation:
        raise ValueError(f"{field}: invalid number") from None
    if not result.is_finite() or result < 0 or result.adjusted() > 100:
        raise ValueError(f"{field}: expected a finite nonnegative number up to 101 digits")
    if result.as_tuple().exponent < -20:
        raise ValueError(f"{field}: maximum 20 decimal places")
    return result


def convert_row(kind: ImportKind, row: dict[str, str], line: int) -> dict:
    if kind == "companies":
        result = {}
        for source, target in COMPANY_FIELDS.items():
            if target == "id":
                result[target] = identifier(row[source], source)
            elif target in BOOLEAN_FIELDS:
                result[target] = boolean(row[source], source)
            elif target == "annual_revenue":
                result[target] = number(row[source], source)
            else:
                result[target] = optional(row[source])
        result["source_data"] = row
        return result
    if kind == "persons":
        return {
            "id": identifier(row["SYNTHETIC_F"], "SYNTHETIC_F"),
            "company_id": identifier(row["SYNTHETIC_U"], "SYNTHETIC_U"),
            "city": optional(row["CITY_NAME"]),
            "region": optional(row["RC_NAME"]),
        }
    amount = number(row["PAY_AMT"], "PAY_AMT", required=True)
    count = number(row["PAY_CNT"], "PAY_CNT", required=True)
    if count != count.to_integral_value() or count > 2**63 - 1:
        raise ValueError("PAY_CNT: expected a nonnegative 64-bit integer")
    return {
        "payer_company_id": identifier(row["PAYER"], "PAYER"),
        "beneficiary_company_id": identifier(row["BENEFICIARY"], "BENEFICIARY"),
        "payment_amount": amount,
        "payment_count": int(count),
        "source_row_number": line,
    }


def parse_csv(
    content: bytes,
    kind: ImportKind,
    encoding: CsvEncoding,
    delimiter: str | None,
    max_rows: int,
) -> ParsedCsv:
    try:
        decoded = content.decode(encoding)
    except UnicodeDecodeError:
        raise CsvValidationError(
            [{"row": 0, "message": f"Invalid {encoding} encoding; try encoding=cp1251"}]
        ) from None
    if "\x00" in decoded:
        raise CsvValidationError([{"row": 0, "message": "NUL bytes are not allowed"}])
    stream = io.StringIO(decoded, newline="")
    if delimiter is None:
        # Only inspect the header: multiline/long article-like values can confuse Sniffer.
        try:
            delimiter = csv.Sniffer().sniff(stream.readline(), delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        stream.seek(0)
    required = {
        "companies": set(COMPANY_FIELDS),
        "persons": PERSON_FIELDS,
        "payments": PAYMENT_FIELDS,
    }[kind]
    errors = []
    rows = []
    companies_seen = set()
    skipped_duplicate_rows = 0
    persons_seen = {}
    person_attribute_conflicts = 0
    try:
        reader = csv.DictReader(stream, delimiter=delimiter, strict=True)
        headers = reader.fieldnames
        if not headers:
            raise CsvValidationError([{"row": 1, "message": "CSV header is missing"}])
        headers = [h.strip().removeprefix("\ufeff") for h in headers]
        reader.fieldnames = headers
        if len(headers) != len(set(headers)) or "" in headers:
            raise CsvValidationError([{"row": 1, "message": "Empty or duplicate column names"}])
        missing = sorted(required - set(headers))
        if missing:
            raise CsvValidationError([{"row": 1, "message": "Missing columns", "columns": missing}])
        for row in reader:
            line = reader.line_num
            if len(rows) >= max_rows:
                raise CsvValidationError([{"row": line, "message": f"Maximum {max_rows} rows"}])
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Number of values does not match header")
                converted = convert_row(kind, row, line)
                if kind == "companies":
                    if converted["id"] in companies_seen:
                        skipped_duplicate_rows += 1
                    companies_seen.add(converted["id"])
                if kind == "persons":
                    person = (converted["city"], converted["region"])
                    previous = persons_seen.get(converted["id"])
                    if previous is not None and previous != person:
                        person_attribute_conflicts += 1
                    persons_seen[converted["id"]] = person
                rows.append(converted)
            except ValueError as exc:
                errors.append({"row": line, "message": str(exc)})
                if len(errors) >= 20:
                    break
    except csv.Error:
        raise CsvValidationError([{"row": reader.line_num, "message": "Malformed CSV"}]) from None
    if errors:
        raise CsvValidationError(errors)
    if not rows:
        raise CsvValidationError([{"row": 2, "message": "CSV has no data rows"}])
    return ParsedCsv(
        rows=rows,
        delimiter=delimiter,
        skipped_duplicate_rows=skipped_duplicate_rows,
        person_attribute_conflicts=person_attribute_conflicts,
    )
