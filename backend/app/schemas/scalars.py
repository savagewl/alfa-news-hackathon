from decimal import Decimal
from typing import Annotated

from pydantic import PlainSerializer, WithJsonSchema


def decimal_string(value: Decimal) -> str:
    """Keep decimal precision and scale without scientific notation in API responses."""
    return format(value, "f")


DecimalString = Annotated[
    Decimal,
    PlainSerializer(decimal_string, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "pattern": r"^-?\d+(?:\.\d+)?$"}, mode="serialization"),
]
