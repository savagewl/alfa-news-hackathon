import json

from app.services.llm.json_stream import JsonFieldDeltaDecoder


def _decoded(chunks: list[str]) -> dict[tuple[str, int | None], str]:
    decoder = JsonFieldDeltaDecoder()
    result: dict[tuple[str, int | None], str] = {}
    for chunk in chunks:
        for delta in decoder.feed(chunk):
            key = (delta["field"], delta.get("index"))
            result[key] = result.get(key, "") + delta["text"]
    return result


def test_decoder_handles_every_split_and_hides_internal_fields():
    raw = (
        r'{"headline":"Риск \"А\"\\Б\n",'
        r'"facts":["Факт \u0411","emoji \uD83D\uDE80"],'
        r'"bank_position":null,"ai_interpretation":"Вывод",'
        r'"claims":[{"text":"НЕ ПОКАЗЫВАТЬ"}],"event_id":"secret"}'
    )
    expected = {
        ("headline", None): 'Риск "А"\\Б\n',
        ("facts", 0): "Факт Б",
        ("facts", 1): "emoji 🚀",
        ("ai_interpretation", None): "Вывод",
    }
    assert _decoded(list(raw)) == expected
    for split in range(len(raw) + 1):
        assert _decoded([raw[:split], raw[split:]]) == expected


def test_decoder_matches_valid_json_values():
    value = {
        "risks": ["Спрос снизится", "Курс изменится"],
        "opportunities": ["Новый рынок"],
        "suggested_actions": ["Проверить сценарий"],
        "limitations": ["Нет прогноза"],
    }
    raw = json.dumps(value, ensure_ascii=True)
    decoded = _decoded([raw[index : index + 3] for index in range(0, len(raw), 3)])
    assert decoded[("risks", 0)] == value["risks"][0]
    assert decoded[("risks", 1)] == value["risks"][1]
    assert decoded[("opportunities", 0)] == value["opportunities"][0]
    assert decoded[("suggested_actions", 0)] == value["suggested_actions"][0]
    assert decoded[("limitations", 0)] == value["limitations"][0]
