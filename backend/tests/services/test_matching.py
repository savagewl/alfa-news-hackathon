from app.services.llm.matching import score_relevance
from app.services.llm.schemas import (
    CompanyFields,
    CompanyProfile,
    CompanyProfileText,
    Counterparty,
    NewsAnalysis,
    OkvedSector,
    ProcessedNews,
)


def _news(**markup_overrides) -> ProcessedNews:
    markup = NewsAnalysis(
        summary="Банк России сохранил ключевую ставку на 14%.",
        topics=["ставка"],
        affected_sectors=[
            OkvedSector(code="25.11", name="Производство строительных металлических конструкций")
        ],
        regions=["Курганская область"],
        impact_kinds=["rate"],
        facts=["Ставка сохранена на 14%"],
        size_relevance=["средний"],
        **markup_overrides,
    )
    return ProcessedNews(
        url="https://example.ru/rate",
        text="Совет директоров Банка России сохранил ключевую ставку на уровне 14%.",
        summary=markup.summary,
        markup=markup,
        vector=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        embedding_model_version="fake",
    )


def _profile(vector: list[float]) -> CompanyProfile:
    text = CompanyProfileText(synthetic_u="U_87", main_text="Компания U_87", aspects={})
    return CompanyProfile(text=text, vectors={"main": vector}, embedding_model_version="fake")


def _fields(**overrides) -> CompanyFields:
    base = dict(
        synthetic_u="U_87",
        okved_code="25.11",
        okved_name="Производство строительных металлических конструкций",
        workers_fns="101-250",
        region="Курганская область",
        city="г. Курган",
        products=["loa"],
    )
    base.update(overrides)
    return CompanyFields(**base)


def test_sector_region_rate_and_scale_match():
    score = score_relevance(_profile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), _fields(), _news())
    matched = {rule.name for rule in score.rules if rule.matched}
    assert {"sector", "region", "rate_products", "scale"} <= matched
    assert score.cosine == 0.0
    assert score.score >= 0.7
    assert score.is_relevant
    assert score.reasons


def test_news_id_passthrough_and_default():
    plain = score_relevance(_profile([0.0] * 8), _fields(), _news())
    assert plain.news_id is None
    linked = score_relevance(
        _profile([0.0] * 8), _fields(), _news(), news_id="11111111-1111-1111-1111-111111111111"
    )
    assert linked.news_id == "11111111-1111-1111-1111-111111111111"
    assert linked.news_url == "https://example.ru/rate"


def test_cosine_contributes_to_score():
    score = score_relevance(
        _profile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _fields(
            okved_code=None,
            okved_name=None,
            region=None,
            city=None,
            workers_fns=None,
            products=[],
        ),
        _news(),
    )
    assert score.cosine == 1.0
    assert not any(rule.matched for rule in score.rules)
    assert 0.39 <= score.score <= 0.4


def test_sector_exact_subclass_beats_class():
    score = score_relevance(_profile([0.0] * 8), _fields(), _news())
    sector = next(rule for rule in score.rules if rule.name == "sector")
    assert sector.weight == 0.4
    assert "подкласса" in sector.detail
    assert "25.11" in sector.detail


def test_sector_class_match_when_subclass_differs():
    news = _news()
    sectors = OkvedSector(code="25.10", name="Прочее производство металлоконструкций")
    markup = news.markup.model_copy(update={"affected_sectors": [sectors]})
    news = news.model_copy(update={"markup": markup})
    score = score_relevance(_profile([0.0] * 8), _fields(), news)
    sector = next(rule for rule in score.rules if rule.name == "sector")
    assert sector.weight == 0.35
    assert "группы" in sector.detail and "25.1" in sector.detail


def test_semantic_similarity_always_in_reasons():
    score = score_relevance(_profile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), _fields(), _news())
    assert any("Семантическое сходство" in reason for reason in score.reasons)


def test_counterparty_match():
    fields = _fields(
        counterparties=[
            Counterparty(name="Совет директоров", direction="outgoing", pay_amt=1, pay_cnt=1)
        ]
    )
    score = score_relevance(_profile([0.0] * 8), fields, _news())
    matched = [rule for rule in score.rules if rule.name == "counterparty" and rule.matched]
    assert matched
    assert "контрагент" in matched[0].detail.lower()


def test_no_match_is_not_relevant():
    fields = CompanyFields(
        synthetic_u="U_9",
        okved_code="46.51",
        okved_name="Торговля оптовая компьютерами",
        workers_fns="Менее 15",
    )
    score = score_relevance(_profile([0.0] * 8), fields, _news())
    assert score.score == 0.0
    assert not score.is_relevant
    assert not any(rule.matched for rule in score.rules)
    assert score.reasons == ["Семантическое сходство: 0.00"]


def test_default_any_size_is_not_a_scale_match():
    markup = _news().markup.model_copy(update={"size_relevance": ["любой"]})
    news = _news().model_copy(update={"markup": markup})
    score = score_relevance(_profile([0.0] * 8), _fields(), news)
    scale = next(rule for rule in score.rules if rule.name == "scale")
    assert not scale.matched


def _news_with_text(text: str) -> ProcessedNews:
    return _news().model_copy(update={"text": text, "summary": text})


def test_product_keywords_respect_word_boundaries():
    fields = _fields(products=["rko", "dep"])
    noise = _news_with_text("Ярко выраженный рост: компания внёс вклад в маркой отрасли.")
    assert not _rule(score_relevance(_profile([0.0] * 8), fields, noise), "product").matched
    real = _news_with_text("Банки повысили ставки по депозитам и тарифы на эквайринг.")
    assert _rule(score_relevance(_profile([0.0] * 8), fields, real), "product").matched


def test_counterparty_requires_whole_word():
    fields = _fields(counterparties=[Counterparty(name="U_1", pay_amt=1, pay_cnt=5)])
    news = _news_with_text("Компания U_10 объявила о слиянии.")
    assert not _rule(score_relevance(_profile([0.0] * 8), fields, news), "counterparty").matched
    news = _news_with_text("Компания «U_1» объявила о слиянии.")
    assert _rule(score_relevance(_profile([0.0] * 8), fields, news), "counterparty").matched


def test_counterparty_ignores_legal_form_and_short_names():
    fields = _fields(
        counterparties=[
            Counterparty(name='ООО "Ромашка"', pay_amt=1, pay_cnt=5),
            Counterparty(name="ИП", pay_amt=1, pay_cnt=5),
        ]
    )
    news = _news_with_text("Поставщик Ромашка расширяет производство.")
    assert _rule(score_relevance(_profile([0.0] * 8), fields, news), "counterparty").matched
    news = _news_with_text("ИП подали заявки.")
    assert not _rule(score_relevance(_profile([0.0] * 8), fields, news), "counterparty").matched


def test_region_stems_match_city_and_oblast_forms():
    markup = _news().markup.model_copy(update={"regions": ["Москва"]})
    news = _news().model_copy(update={"markup": markup})
    fields = _fields(region="Московская область", city="г. Москва")
    assert _rule(score_relevance(_profile([0.0] * 8), fields, news), "region").matched

    markup = _news().markup.model_copy(update={"regions": ["Курган"]})
    news = _news().model_copy(update={"markup": markup})
    fields = _fields(region="Курганская область", city=None)
    assert _rule(score_relevance(_profile([0.0] * 8), fields, news), "region").matched


def _rule(score, name):
    return next(rule for rule in score.rules if rule.name == name)


def test_region_only_news_is_relevant_with_typical_cosine():
    """Региональная новость без других совпадений проходит порог при типичном cosine bge-m3."""
    markup = _news().markup.model_copy(
        update={
            "regions": ["Москва"],
            "affected_sectors": [],
            "impact_kinds": ["other"],
            "size_relevance": ["любой"],
        }
    )
    news = _news().model_copy(update={"markup": markup})
    fields = _fields(
        region="Москва", city="г. Москва", okved_code=None, okved_name=None, products=[]
    )
    # cosine ≈ 0.45: 0.4 * 0.45 + 0.2 (регион) = 0.38 >= порог 0.35
    vector = [0.45, (1 - 0.45**2) ** 0.5, 0, 0, 0, 0, 0, 0]
    score = score_relevance(_profile(vector), fields, news)
    region = next(rule for rule in score.rules if rule.name == "region")
    assert region.matched and region.weight == 0.2
    assert score.is_relevant


def test_sector_match_by_class_only_is_weak_and_not_relevant_alone():
    """Класс 25 включает и металлоконструкции (25.11), и оружие (25.4)."""

    markup = _news().markup.model_copy(
        update={
            "affected_sectors": [OkvedSector(code="25.40", name="Производство оружия")],
            "regions": [],
            "impact_kinds": ["other"],
            "size_relevance": ["любой"],
        }
    )
    news = _news().model_copy(update={"markup": markup})
    fields = _fields(region=None, city=None, workers_fns=None, products=[], okved_name=None)
    score = score_relevance(_profile([0.0] * 8), fields, news)
    sector = next(rule for rule in score.rules if rule.name == "sector")
    assert sector.matched and sector.weight == 0.15 and "Смежная" in sector.detail
    assert not score.is_relevant


def test_sector_class_level_news_markup_matches_whole_class():
    markup = _news().markup.model_copy(
        update={"affected_sectors": [OkvedSector(code="25", name="Металлообработка")]}
    )
    news = _news().model_copy(update={"markup": markup})
    score = score_relevance(_profile([0.0] * 8), _fields(), news)
    sector = next(rule for rule in score.rules if rule.name == "sector")
    assert sector.weight == 0.35 and "класса" in sector.detail


def test_banking_signals_alone_are_capped_without_sector_region_or_counterparty():
    markup = _news().markup.model_copy(
        update={
            "affected_sectors": [],
            "regions": [],
            "impact_kinds": ["rate"],
            "size_relevance": ["любой"],
        }
    )
    news = _news().model_copy(
        update={
            "markup": markup,
            "text": "Банк запустил льготные кредиты для продавцов маркетплейса.",
            "summary": "Льготные кредиты для продавцов маркетплейса.",
        }
    )
    fields = _fields(
        okved_code=None, okved_name=None, region=None, city=None, workers_fns=None, products=["loa"]
    )
    # cosine 0.35: 0.4 * 0.35 + кредитование 0.15 + ставка 0.15 = 0.44 без ограничения
    vector = [(1 - 0.35**2) ** 0.5, 0.35, 0, 0, 0, 0, 0, 0]
    score = score_relevance(_profile(vector), fields, news)
    assert {rule.name for rule in score.rules if rule.matched} == {"product", "rate_products"}
    assert score.score < 0.35 and not score.is_relevant
    # с совпавшей отраслью ограничения нет
    with_sector = _news().model_copy(update={"text": news.text, "summary": news.summary})
    sector_fields = _fields(region=None, city=None, workers_fns=None, products=["loa"])
    assert score_relevance(_profile(vector), sector_fields, with_sector).is_relevant


def test_related_persons_region_makes_news_relevant_when_company_region_differs():
    from app.services.llm.schemas import PersonGeo

    markup = _news().markup.model_copy(
        update={
            "regions": ["Тюменская область"],
            "affected_sectors": [],
            "impact_kinds": ["other"],
            "size_relevance": ["любой"],
        }
    )
    news = _news().model_copy(update={"markup": markup})
    fields = _fields(
        okved_code=None,
        okved_name=None,
        workers_fns=None,
        products=[],
        persons_total=2,
        persons=[PersonGeo(city="Тюмень", region="Тюменская область", count=2)],
    )
    score = score_relevance(_profile([0.0] * 8), fields, news)
    persons = next(rule for rule in score.rules if rule.name == "persons")
    region = next(rule for rule in score.rules if rule.name == "region")
    assert persons.matched and not region.matched
    assert "связанных физлиц" in persons.detail and "Тюмен" in persons.detail
    assert any("физлиц" in reason for reason in score.reasons)

    # регион самой компании уже даёт баллы — правило физлиц не дублирует их
    same_region = _fields(
        region="Тюменская область",
        city=None,
        persons_total=1,
        persons=[PersonGeo(region="Тюменская область")],
    )
    both = score_relevance(_profile([0.0] * 8), same_region, news)
    assert next(r for r in both.rules if r.name == "region").matched
    assert not next(r for r in both.rules if r.name == "persons").matched

    # без физлиц правило молчит
    none = score_relevance(_profile([0.0] * 8), _fields(), news)
    assert not next(r for r in none.rules if r.name == "persons").matched


def test_place_stems_find_declined_names_in_text():
    from app.services.llm.matching import place_stems

    assert place_stems("Тюмень") == {"тюмен"} == place_stems("Тюменская область")
    assert "москв" in place_stems("Москва") and "моско" in place_stems("Московская область")
    text = "в москве открыли завод"
    assert any(stem in text for stem in place_stems("г. Москва"))
