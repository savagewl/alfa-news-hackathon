"""Incrementally expose safe, user-facing fields from a streamed JSON object."""

from __future__ import annotations

ROOT_STRING_FIELDS = frozenset({"headline", "ai_interpretation", "bank_position"})
ROOT_LIST_FIELDS = frozenset(
    {"facts", "risks", "opportunities", "suggested_actions", "limitations"}
)


class JsonFieldDeltaDecoder:
    """Stateful JSON lexer which never exposes nested/internal fields such as claims."""

    def __init__(self) -> None:
        self._stack: list[str] = []
        self._in_string = False
        self._token_kind = ""
        self._key_chars: list[str] = []
        self._root_key: str | None = None
        self._expecting_root_key = False
        self._list_field: str | None = None
        self._list_index = 0
        self._target_field: str | None = None
        self._target_index: int | None = None
        self._escaped = False
        self._unicode_digits: list[str] | None = None
        self._high_surrogate: int | None = None

    def feed(self, fragment: str) -> list[dict]:
        emitted: list[dict] = []
        for char in fragment:
            if self._in_string:
                self._feed_string_char(char, emitted)
            else:
                self._feed_structure_char(char)
        return emitted

    def _feed_structure_char(self, char: str) -> None:
        if char == '"':
            self._begin_string()
        elif char == "{":
            self._stack.append(char)
            if len(self._stack) == 1:
                self._expecting_root_key = True
        elif char == "[":
            if self._stack == ["{"] and self._root_key in ROOT_LIST_FIELDS:
                self._list_field = self._root_key
                self._list_index = 0
            self._stack.append(char)
        elif char == "]":
            if self._stack == ["{", "["]:
                self._list_field = None
            if self._stack:
                self._stack.pop()
        elif char == "}":
            if self._stack:
                self._stack.pop()
        elif char == ":" and self._stack == ["{"]:
            self._expecting_root_key = False
        elif char == "," and self._stack == ["{"]:
            self._root_key = None
            self._expecting_root_key = True

    def _begin_string(self) -> None:
        self._in_string = True
        self._escaped = False
        self._unicode_digits = None
        self._high_surrogate = None
        self._key_chars = []
        self._target_field = None
        self._target_index = None
        if self._stack == ["{"] and self._expecting_root_key:
            self._token_kind = "key"
        elif self._stack == ["{"] and self._root_key in ROOT_STRING_FIELDS:
            self._token_kind = "value"
            self._target_field = self._root_key
        elif self._stack == ["{", "["] and self._list_field is not None:
            self._token_kind = "value"
            self._target_field = self._list_field
            self._target_index = self._list_index
        else:
            self._token_kind = "ignored"

    def _feed_string_char(self, char: str, emitted: list[dict]) -> None:
        if self._unicode_digits is not None:
            self._unicode_digits.append(char)
            if len(self._unicode_digits) == 4:
                digits = "".join(self._unicode_digits)
                self._unicode_digits = None
                try:
                    self._accept_code_unit(int(digits, 16), emitted)
                except ValueError:
                    self._high_surrogate = None
            return
        if self._escaped:
            self._escaped = False
            if char == "u":
                self._unicode_digits = []
                return
            decoded = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }.get(char)
            if decoded is not None:
                self._accept_decoded(decoded, emitted)
            return
        if char == "\\":
            self._escaped = True
            return
        if char == '"':
            self._finish_string()
            return
        self._accept_decoded(char, emitted)

    def _accept_code_unit(self, value: int, emitted: list[dict]) -> None:
        if 0xD800 <= value <= 0xDBFF:
            self._high_surrogate = value
            return
        if 0xDC00 <= value <= 0xDFFF and self._high_surrogate is not None:
            codepoint = 0x10000 + ((self._high_surrogate - 0xD800) << 10) + (value - 0xDC00)
            self._high_surrogate = None
            self._accept_decoded(chr(codepoint), emitted)
            return
        self._high_surrogate = None
        if not 0xD800 <= value <= 0xDFFF:
            self._accept_decoded(chr(value), emitted)

    def _accept_decoded(self, text: str, emitted: list[dict]) -> None:
        if self._high_surrogate is not None:
            self._high_surrogate = None
        if self._token_kind == "key":
            self._key_chars.append(text)
            return
        if self._token_kind != "value" or self._target_field is None:
            return
        item = {"field": self._target_field, "text": text}
        if self._target_index is not None:
            item["index"] = self._target_index
        if emitted and all(
            emitted[-1].get(key) == item.get(key) for key in ("field", "index")
        ):
            emitted[-1]["text"] += text
        else:
            emitted.append(item)

    def _finish_string(self) -> None:
        if self._token_kind == "key":
            self._root_key = "".join(self._key_chars)
        elif self._target_index is not None:
            self._list_index += 1
        self._in_string = False
        self._token_kind = ""
        self._target_field = None
        self._target_index = None
        self._unicode_digits = None
        self._high_surrogate = None
