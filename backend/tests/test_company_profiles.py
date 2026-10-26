from types import SimpleNamespace

from app.repositories.company_profiles import build_company_profile_input, company_fields
from app.services.llm.config import ServicesSettings
from app.services.llm.schemas import Counterparty


def _company(**changes):
    values = {
        "id": "company-1",
        "org_type": "ООО",
        "ownership_form": "частная",
        "age_category": "6-10",
        "development_stage": "рост",
        "okved_code": "62.01",
        "okved_name": "Разработка ПО",
        "key_owner_flag": False,
        "employees_fns_category": "16-100",
        "employees_spark_category": None,
        "client_type": "средний",
        "has_rko": True,
        "has_guarantees": False,
        "has_credit": True,
        "has_leasing": None,
        "has_deposit": False,
        "has_payroll": True,
        "has_fts_data": False,
        "fts_country_name": None,
        "has_factoring": False,
        "annual_revenue": None,
        "group_id": None,
        "city": "Казань",
        "region": "Татарстан",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_company_fields_preserve_known_false_and_products():
    fields = company_fields(_company(), [])
    assert fields.negative_mark == "0"
    assert fields.products == ["rko", "loa", "zpp"]

    unknown = company_fields(_company(key_owner_flag=None), [])
    assert unknown.negative_mark is None


def test_profile_fingerprint_changes_with_counterparty_and_model():
    counterparties = [Counterparty(name="Поставщик", pay_amt=100, pay_cnt=2)]
    settings = ServicesSettings(_env_file=None, embed_model="bge-m3")
    initial = build_company_profile_input(_company(), counterparties, settings)
    repeated = build_company_profile_input(_company(), counterparties, settings)
    assert initial.fingerprint == repeated.fingerprint
    assert initial.embedding_model_version == "bge-m3:1024d"

    changed = [Counterparty(name="Поставщик", pay_amt=101, pay_cnt=2)]
    changed_counterparty = build_company_profile_input(_company(), changed, settings)
    changed_model = build_company_profile_input(
        _company(),
        changed,
        settings.model_copy(update={"embed_model": "bge-m3-v2"}),
    )
    assert changed_counterparty.fingerprint != initial.fingerprint
    assert changed_model.fingerprint != changed_counterparty.fingerprint


def test_related_persons_enter_profile_text_and_change_fingerprint_only_when_present():
    from app.services.llm.schemas import PersonGeo

    settings = ServicesSettings(_env_file=None)
    base = build_company_profile_input(_company(), [], settings)
    assert "физлица" not in base.text.main_text
    same = build_company_profile_input(_company(), [], settings, (0, []))
    assert same.fingerprint == base.fingerprint  # кэш профилей без физлиц не сбрасывается

    persons = (
        4,
        [
            PersonGeo(city="Москва", region="Москва", count=3),
            PersonGeo(region="Курганская область"),
        ],
    )
    with_persons = build_company_profile_input(_company(), [], settings, persons)
    assert "Связанные физлица: 4 (Москва, Москва — 3; Курганская область — 1)" in (
        with_persons.text.main_text
    )
    assert "связанные физлица: 4" in with_persons.text.aspects["industry"]
    assert with_persons.fingerprint != base.fingerprint
    assert with_persons.fields.persons_total == 4
