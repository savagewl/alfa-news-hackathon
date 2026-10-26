from app.services.llm.company_profile import (
    build_company_profile,
    build_company_profile_text,
    format_revenue,
    profile_field_labels,
)
from app.services.llm.schemas import CompanyFields, Counterparty

CSV_ROW = {
    "SYNTHETIC_U": "U_67",
    "ORG_TYPE": "КОММ_ОРГ",
    "OKFS_NAME": "ЧАСТНАЯ СОБСТВЕННОСТЬ",
    "COMPANY_AGE_YEARS_GRADATION": "16-25",
    "ECO_MARK_LABEL": "Рождение",
    "MAINOKVED_CODE": "46.51",
    "MAINOKVED_NAME": "Торговля оптовая компьютерами",
    "NEGATIVEMARK": "0.0",
    "AMOUNT_WORKERS_FNS_GRADATION": "Менее 15",
    "AMOUNT_WORKERS_SPARK_GRADATION": "",
    "CL_TYPE": "PROSPECT",
    "CL_PRODUCT_RKO_MARK": "0.0",
    "CL_PRODUCT_GAR_MARK": "0.0",
    "CL_PRODUCT_LOA_MARK": "0.0",
    "CL_PRODUCT_LEA_MARK": "1.0",
    "CL_PRODUCT_DEP_MARK": "0.0",
    "CL_PRODUCT_ZPP_MARK": "0.0",
    "FTS_FLAG": "",
    "FTS_COUNTRY_NAME": "",
    "FACTORING_MARK": "1.0",
    "REVENUE_COMPANY": "136493.0",
    "GROUP_ID": "group_134463_sb",
    "CITY": "г. Новосибирск",
    "REGION": "Новосибирская область",
}


def test_from_csv_row():
    fields = CompanyFields.from_csv_row(CSV_ROW)
    assert fields.synthetic_u == "U_67"
    assert fields.products == ["lea"]
    assert fields.factoring is True
    assert fields.fts_flag is False
    assert fields.revenue == 136493.0
    assert fields.workers_spark is None


def test_profile_text_unknowns():
    fields = CompanyFields(synthetic_u="U_1")
    profile = build_company_profile_text(fields)
    assert "неизвестно" in profile.main_text
    assert set(profile.aspects) == {"industry", "finance", "ved"}
    assert profile.aspects["ved"] == "Экспортная деятельность: нет; отрасль: неизвестно"


def test_profile_text_from_csv_row():
    fields = CompanyFields.from_csv_row(CSV_ROW)
    profile = build_company_profile_text(fields)
    assert "Отрасль (ОКВЭД 46.51): Торговля оптовая компьютерами" in profile.main_text
    assert "лизинг" in profile.main_text
    assert "Выручка: 136 493 руб." in profile.main_text
    assert "ВЭД: нет" in profile.main_text
    assert "Негативная метка: нет" in profile.main_text


def test_profile_text_negative_mark():
    fields = CompanyFields(synthetic_u="U_3", negative_mark="1.0")
    profile = build_company_profile_text(fields)
    assert "Негативная метка: есть (причина неизвестна)" in profile.main_text


def test_profile_field_labels_conditional():
    fields = CompanyFields.from_csv_row(CSV_ROW)
    labels = profile_field_labels(fields)
    assert "Отрасль (ОКВЭД 46.51)" in labels
    assert "Группа компаний" in labels
    assert "Контрагенты" not in labels  # в CSV_ROW контрагентов нет
    assert "Регион" not in labels  # в тексте профиля такого поля нет, только «География»

    bare = profile_field_labels(CompanyFields(synthetic_u="U_1"))
    assert "Отрасль" in bare
    assert "Группа компаний" not in bare
    assert "Контрагенты" not in bare


def test_profile_with_counterparties(fake_embedder):
    fields = CompanyFields(
        synthetic_u="U_2",
        products=["rko"],
        workers_fns="16-100",
        counterparties=[
            Counterparty(name="U_9", direction="incoming", pay_amt=2_748_000, pay_cnt=4),
            Counterparty(name="U_10", direction="outgoing", pay_amt=16_157_000, pay_cnt=981),
        ],
    )
    profile = build_company_profile(fields, fields.counterparties, embedder=fake_embedder)
    assert "Контрагенты: U_10" in profile.text.main_text
    assert "входящие" in profile.text.main_text
    assert set(profile.vectors) == {"main", "industry", "finance", "ved"}
    assert all(len(vector) == 8 for vector in profile.vectors.values())
    assert profile.embedding_model_version == "fake-embedder:8d"
    assert fake_embedder.calls == [[profile.text.main_text, *profile.text.aspects.values()]]


def test_format_revenue():
    assert format_revenue(136493) == "136 493 руб."
    assert format_revenue(1_400_000_000) == "1.4 млрд руб."
    assert format_revenue(15_000_000_000) == "15.0 млрд руб."
    assert format_revenue(None) == "неизвестно"


def test_profile_uses_counterparties_from_fields_and_labels_match():
    fields = CompanyFields(
        synthetic_u="U_5",
        counterparties=[Counterparty(name="U_8", direction="incoming", pay_amt=1000, pay_cnt=4)],
    )
    profile = build_company_profile_text(fields)
    assert "Контрагенты: U_8" in profile.main_text
    assert "Контрагенты" in profile_field_labels(fields)
    assert "Контрагенты" in profile_field_labels(
        CompanyFields(synthetic_u="U_6"), [Counterparty(name="U_9")]
    )
