import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# «Прямых рисков нет», «не несёт значимых возможностей», «Не выявлено» — это отсутствие
# данных, а не оценка.
_NOTHING = r"(?:(?:прямых|значимых|существенных|явных)\s+)?(?:рисков|возможностей)"
_NOT_FOUND = r"не\s+(?:выявлен\w*|обнаружен\w*|найден\w*|прослежива\w*|просматрива\w*)"
_FOR_COMPANY = r"(?:для\s+(?:компании|бизнеса)\s+)?"
_EMPTY_STATEMENT = re.compile(
    rf"^(?:(?:для компании|для бизнеса|в данном случае|по данной новости)\s*,?\s*)?"
    rf"(?:(?:нет|отсутствуют)\s+{_NOTHING}"
    rf"|(?:новость|событие)?\s*не\s+(?:несет|несёт|создает|создаёт|даёт|дает|содержит)\s+{_NOTHING}"
    rf"|{_NOTHING}(?:\s+и\s+{_NOTHING})?\s+{_FOR_COMPANY}(?:нет|отсутствуют|{_NOT_FOUND})"
    rf"|{_NOT_FOUND})",
    re.IGNORECASE,
)
_EMPTY_WORDS = {"нет", "null", "none", "n/a", "-", "—"}


def shorten_text(text: str, *, sentences: int, chars: int) -> str:
    """Не больше `sentences` предложений и `chars` символов (обрезка по слову, с «…»)."""

    parts = _SENTENCE_END.split(" ".join(text.split()))
    result = " ".join(parts[:sentences])
    if len(result) > chars:
        result = result[:chars].rsplit(" ", 1)[0].rstrip(",;:—- ") + "…"
    return result


def is_empty_assessment(text: str | None) -> bool:
    """Пустой блок оценки: None, пробелы, «нет», «null» или фраза вида «рисков нет»."""

    value = " ".join((text or "").split()).strip(" .")
    if not value or value.lower() in _EMPTY_WORDS:
        return True
    first = _SENTENCE_END.split(value)[0].strip(" .,")
    return len(first) < 90 and bool(_EMPTY_STATEMENT.match(first))
