import difflib
from collections.abc import Callable

from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.json_stream import JsonFieldDeltaDecoder
from app.services.llm.provider import (
    LLMProvider,
    complete_json,
    complete_text,
    get_llm_provider,
    messages,
)
from app.services.llm.schemas import (
    BankPassage,
    CompanyContext,
    InsightClaim,
    InsightDraft,
    InsightResult,
    InsightStep,
    NewsEvent,
    Verification,
)

ANALYST_SYSTEM_PROMPT = (
    "Ты — финансовый аналитик Альфа-Банка, готовишь персонализированный разбор события "
    "для крупного бизнеса. Верни строго JSON без markdown:\n"
    "{\n"
    '  "event_id": str | null,\n'
    '  "company_id": str | null,\n'
    '  "headline": "заголовок инсайта",\n'
    '  "facts": ["что произошло: факты, участники, цифры, сроки"],\n'
    '  "bank_position": "позиция/прогноз аналитиков Альфа-Банка со ссылкой [B#] или null",\n'
    '  "ai_interpretation": "почему это важно именно для этой компании: связь с отраслью, '
    'продуктами, регионами, контрагентами",\n'
    '  "risks": ["риски для компании"],\n'
    '  "opportunities": ["возможности для компании"],\n'
    '  "suggested_actions": ["конкретные действия"],\n'
    '  "limitations": ["что нельзя уверенно оценить"],\n'
    '  "claims": [{"id": "c1", "text": "...", "kind": "fact|bank_position|ai_inference", '
    '"evidence_refs": ["news:<url>", "bank:<document_id>#<chunk_id>", "company:<поле>"]}]\n'
    "}\n"
    "Правила: каждое ключевое утверждение должно иметь обоснование в claims с evidence_refs; "
    "bank_position — только по блоку «Аналитика Альфа-Банка», "
    "вид вывода AI не подменяй мнением банка; "
    "не выдумывай цифры; если данных мало — всё равно дай черновик и перечисли ограничения. "
    "claims — не более 6, только существенные для вывода; "
    "evidence_refs вида company:<поле> — используй только дословные названия полей "
    "из блока «=== ПРОФИЛЬ КОМПАНИИ ===» (например company:Выручка, company:Контрагенты); "
    "метку отрасли пиши дословно из профиля: company:Отрасль (ОКВЭД 35.12), а не company:Отрасль; "
    "если у компании есть негативная метка — упомяни в risks необходимость проверить её "
    "причину, не утверждай дефолт или просрочку без данных; "
    "операции контрагента с числом платежей < 3 считай разовыми: не делай из них выводов "
    "о системной ликвидности или концентрации и отрази это в limitations. "
    "Если новость малоинформативна — пиши кратко, без раздувания списков."
)

VERIFIER_SYSTEM_PROMPT = (
    "Ты — проверяющий аналитик (verifier). Проверь черновик инсайта по правилам:\n"
    "1. facts не искажают текст новости.\n"
    "2. bank_position опирается на существующий passage [B#] и не подменяет вывод AI.\n"
    "3. каждый claim имеет evidence_refs из доступных источников; ссылки корректны: "
    "company:<поле> должно дословно существовать в блоке «=== ПРОФИЛЬ КОМПАНИИ ===».\n"
    "4. разделы risks/opportunities/suggested_actions заполнены, если данных достаточно.\n"
    "5. limitations указывают на недостающие данные.\n"
    "6. выводы о ликвидности, концентрации или долговой нагрузке запрещены, если они "
    "опираются на контрагентские операции с числом платежей < 3 (разовые операции).\n"
    "Верни строго JSON:\n"
    '{"status": "APPROVED", "remarks": ["замечания"], "missing_evidence": ["что не подтверждено"], '
    '"retry_queries": ["поисковые запросы для добора данных"]}\n'
    "status: APPROVED — можно публиковать; REVISE — доработать; INSUFFICIENT — данных не хватает. "
    "При APPROVED remarks оставляй пустым: замечания нужны только для REVISE/INSUFFICIENT."
)

REVISE_SYSTEM_PROMPT = (
    "Ты — финансовый аналитик Альфа-Банка. Переработай черновик инсайта с учётом замечаний "
    "проверяющего: исправь подтверждённые замечания, добери обоснования из доступных блоков, "
    "недоступное перенеси в limitations. Не увеличивай количество claims. "
    "Верни JSON той же схемы, что и исходный черновик."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "Ты — консультант крупного бизнеса в интернет-банке «Альфа-Бизнес». Отвечай по-русски, "
    "структурировано, опираясь только на предоставленный контекст: профиль компании, инсайт, "
    "новость и аналитику Альфа-Банка [B#]. Если данных недостаточно — скажи прямо. "
    "Не выдумывай цифры и не подменяй мнение банка собственными выводами. "
    "Операции контрагента с числом платежей < 3 — разовые: не строй на них выводов "
    "о регулярности потоков или ликвидности. "
    "Если даны «Другие релевантные новости» — выбери те, что ближе всего к вопросу "
    "пользователя, используй их и указывай их источники."
)


def analyze_event(
    company_context: CompanyContext,
    news: NewsEvent,
    bank_passages: list[BankPassage],
    *,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
    content_callback: Callable[[str], None] | None = None,
    attempt_callback: Callable[[], None] | None = None,
) -> tuple[InsightDraft, InsightStep]:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    user = (
        f"{_context_block(company_context)}\n\n{_news_block(news)}\n\n"
        f"{_passages_block(bank_passages)}\n\nСоставь персонализированный разбор события."
    )
    draft, response = complete_json(
        "insight_analyst",
        InsightDraft,
        messages(ANALYST_SYSTEM_PROMPT, user),
        provider=llm,
        settings=settings,
        content_callback=content_callback,
        attempt_callback=attempt_callback,
    )
    draft = draft.model_copy(
        update={
            "event_id": draft.event_id or news.url,
            "company_id": draft.company_id or company_context.company_id,
        }
    )
    draft, refs_note = _normalize_evidence_refs(draft, company_context.evidence_labels or None)
    detail = f"headline: {draft.headline}"
    if refs_note:
        detail += f"; evidence_refs: {refs_note}"
    step = InsightStep(
        step="analyst",
        status="ok",
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        detail=detail,
    )
    return draft, step


def verify_insight(
    draft: InsightDraft,
    company_context: CompanyContext,
    news: NewsEvent,
    bank_passages: list[BankPassage],
    *,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
) -> tuple[Verification, InsightStep]:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    user = (
        f"{_context_block(company_context)}\n\n{_news_block(news)}\n\n"
        f"{_passages_block(bank_passages)}\n\n{_draft_block(draft)}\n\nПроверь черновик."
    )
    verification, response = complete_json(
        "insight_verifier",
        Verification,
        messages(VERIFIER_SYSTEM_PROMPT, user),
        provider=llm,
        settings=settings,
    )
    step = InsightStep(
        step="verifier",
        status=verification.status,
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        detail=f"замечаний: {len(verification.remarks)}",
    )
    return verification, step


def revise_insight(
    draft: InsightDraft,
    verification: Verification,
    company_context: CompanyContext,
    news: NewsEvent,
    bank_passages: list[BankPassage],
    *,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
    content_callback: Callable[[str], None] | None = None,
    attempt_callback: Callable[[], None] | None = None,
) -> tuple[InsightDraft, InsightStep]:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    user = (
        f"{_context_block(company_context)}\n\n{_news_block(news)}\n\n"
        f"{_passages_block(bank_passages)}\n\n{_draft_block(draft)}\n\n"
        f"{_verification_block(verification)}\n\nПереработай черновик."
    )
    revised, response = complete_json(
        "insight_revise",
        InsightDraft,
        messages(REVISE_SYSTEM_PROMPT, user),
        provider=llm,
        settings=settings,
        content_callback=content_callback,
        attempt_callback=attempt_callback,
    )
    revised = revised.model_copy(
        update={"event_id": draft.event_id, "company_id": draft.company_id}
    )
    revised, refs_note = _normalize_evidence_refs(revised, company_context.evidence_labels or None)
    detail = f"по замечаниям: {len(verification.remarks)}"
    if refs_note:
        detail += f"; evidence_refs: {refs_note}"
    step = InsightStep(
        step="revise",
        status="ok",
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        detail=detail,
    )
    return revised, step


def build_insight(
    company_context: CompanyContext,
    news: NewsEvent,
    bank_passages: list[BankPassage],
    *,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
    progress_callback: Callable[[str, dict], None] | None = None,
    revision_start: int = 0,
) -> InsightResult:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    steps: list[InsightStep] = []
    revision_version = revision_start - 1
    decoder: JsonFieldDeltaDecoder | None = None

    def begin_revision() -> None:
        nonlocal revision_version, decoder
        revision_version += 1
        decoder = JsonFieldDeltaDecoder()
        _emit_progress(
            progress_callback,
            "reset",
            {"revision_version": revision_version, "verified": False},
        )

    def emit_content(fragment: str) -> None:
        if decoder is None:
            return
        for delta in decoder.feed(fragment):
            _emit_progress(
                progress_callback,
                "delta",
                {**delta, "revision_version": revision_version, "verified": False},
            )

    stream_content = emit_content if progress_callback is not None else None
    stream_attempt = begin_revision if progress_callback is not None else None
    _emit_progress(progress_callback, "progress", {"stage": "generating"})
    draft, step = analyze_event(
        company_context,
        news,
        bank_passages,
        llm=llm,
        settings=settings,
        content_callback=stream_content,
        attempt_callback=stream_attempt,
    )
    steps.append(step)
    _emit_progress(
        progress_callback,
        "draft",
        {
            "draft": draft.model_dump(mode="json"),
            "revision_version": revision_version,
            "verified": False,
        },
    )
    _emit_progress(progress_callback, "progress", {"stage": "verifying"})
    verification, step = verify_insight(
        draft, company_context, news, bank_passages, llm=llm, settings=settings
    )
    steps.append(step)
    cycles = 0
    while verification.status == "REVISE" and cycles < settings.insight_max_cycles:
        _emit_progress(progress_callback, "progress", {"stage": "revising"})
        draft, step = revise_insight(
            draft,
            verification,
            company_context,
            news,
            bank_passages,
            llm=llm,
            settings=settings,
            content_callback=stream_content,
            attempt_callback=stream_attempt,
        )
        steps.append(step)
        _emit_progress(
            progress_callback,
            "draft",
            {
                "draft": draft.model_dump(mode="json"),
                "revision_version": revision_version,
                "verified": False,
            },
        )
        _emit_progress(progress_callback, "progress", {"stage": "verifying"})
        verification, step = verify_insight(
            draft, company_context, news, bank_passages, llm=llm, settings=settings
        )
        steps.append(step)
        cycles += 1
    return InsightResult(draft=draft, verification=verification, steps=steps)


def _emit_progress(
    callback: Callable[[str, dict], None] | None, event: str, data: dict
) -> None:
    if callback is None:
        return
    try:
        callback(event, data)
    except Exception:
        # UI progress must never change the analytical result or retry behavior.
        return


def answer_question(
    company_context: CompanyContext,
    news: NewsEvent,
    draft: InsightDraft,
    question: str,
    bank_passages: list[BankPassage] = (),
    *,
    related_news: list[NewsEvent] | None = None,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
) -> tuple[str, InsightStep]:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    other_news = [item for item in (related_news or []) if item.url != news.url]
    user = (
        f"{_context_block(company_context)}\n\n{_news_block(news)}\n\n"
        f"{_passages_block(list(bank_passages))}\n\n"
        f"{_related_news_block(other_news)}\n\n"
        f"{_draft_block(draft)}\n\n"
        f"Вопрос пользователя: {question}\n\nДай ответ."
    )
    answer, response = complete_text(
        "followup", messages(FOLLOWUP_SYSTEM_PROMPT, user), provider=llm, settings=settings
    )
    step = InsightStep(
        step="followup",
        status="ok",
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        detail=f"вопрос: {question[:80]}",
    )
    return answer, step


def answer_question_structured(
    company_context: CompanyContext,
    news: NewsEvent,
    draft: InsightDraft,
    question: str,
    response_schema,
    bank_passages: list[BankPassage] = (),
    *,
    history: str | None = None,
    related_news: list[NewsEvent] | None = None,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
):
    """Validated JSON follow-up response using the same bounded source context."""

    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    other_news = [item for item in (related_news or []) if item.url != news.url]
    history_block = f"=== ИСТОРИЯ ДИАЛОГА ===\n{history}\n\n" if history else ""
    user = (
        f"{_context_block(company_context)}\n\n{_news_block(news)}\n\n"
        f"{_passages_block(list(bank_passages))}\n\n"
        f"{_related_news_block(other_news)}\n\n"
        f"{_draft_block(draft)}\n\n{history_block}"
        f"Вопрос пользователя: {question}\n\nДай ответ в запрошенной JSON-схеме."
    )
    value, response = complete_json(
        "followup", response_schema, messages(FOLLOWUP_SYSTEM_PROMPT, user), provider=llm,
        settings=settings,
    )
    step = InsightStep(
        step="followup",
        status="ok",
        tokens_prompt=response.prompt_tokens,
        tokens_completion=response.completion_tokens,
        detail=f"вопрос: {question[:80]}",
    )
    return value, step


def _related_news_block(other_news: list[NewsEvent]) -> str:
    if not other_news:
        return ""
    lines = ["=== ДРУГИЕ РЕЛЕВАНТНЫЕ НОВОСТИ ==="]
    for news in other_news:
        source = f"news:{news.url}" if news.url else "news:без url"
        title = f" «{news.title}»" if news.title else ""
        lines.append(f"- {source}{title} — {news.summary}")
    return "\n".join(lines)


# Дословные метки полей, которые допускаются в evidence_refs вида «company:<поле>».
# Полностью совпадающие метки проходят, близкие исправляются (difflib), неверные — удаляются.
_COMPANY_LABELS = (
    "Тип организации",
    "Форма собственности",
    "Возраст",
    "Стадия развития",
    "Отрасль",
    "Численность сотрудников",
    "Тип клиента",
    "Продукты банка",
    "Выручка",
    "Негативная метка",
    "Факторинг",
    "ВЭД",
    "География",
    "Группа компаний",
    "Контрагенты",
)

_COMPANY_PREFIX = "company:"


def _normalize_evidence_refs(draft: InsightDraft, labels=None) -> tuple[InsightDraft, str | None]:
    allowed_labels = list(labels) if labels else list(_COMPANY_LABELS)
    claims: list[InsightClaim] = []
    fixed, dropped = 0, 0
    fixed_refs: list[str] = []
    dropped_refs: list[str] = []
    for claim in draft.claims:
        refs: list[str] = []
        for ref in claim.evidence_refs:
            if not ref.startswith(_COMPANY_PREFIX):
                refs.append(ref)
                continue
            label = ref[len(_COMPANY_PREFIX) :].strip()
            if label in allowed_labels:
                refs.append(ref)
                continue
            match = difflib.get_close_matches(label, allowed_labels, n=1, cutoff=0.6)
            if match:
                refs.append(f"{_COMPANY_PREFIX}{match[0]}")
                fixed += 1
                fixed_refs.append(f"{label}→{match[0]}")
            else:
                dropped += 1
                dropped_refs.append(label)
        claims.append(claim.model_copy(update={"evidence_refs": refs or claim.evidence_refs}))
    if not fixed and not dropped:
        return draft, None
    parts: list[str] = []
    if fixed:
        parts.append(f"исправлено {fixed}: {', '.join(fixed_refs)}")
    if dropped:
        parts.append(f"удалено {dropped}: {', '.join(dropped_refs)}")
    return draft.model_copy(update={"claims": claims}), "; ".join(parts)


def _context_block(context: CompanyContext) -> str:
    lines = [
        "=== ПРОФИЛЬ КОМПАНИИ ===",
        context.profile_text,
    ]
    if context.evidence_labels:
        allowed = ", ".join(f"company:{label}" for label in context.evidence_labels)
        lines.append(f"Допустимые метки для evidence_refs: {allowed}")
    if context.relevance_reasons:
        lines.append("Причины отбора новости: " + "; ".join(context.relevance_reasons))
    if context.as_of:
        lines.append(f"Актуальность профиля: {context.as_of}")
    return "\n".join(lines)


def _news_block(news: NewsEvent) -> str:
    lines = ["=== НОВОСТЬ ==="]
    if news.url:
        lines.append(f"Источник: news:{news.url}")
    if news.title:
        lines.append(f"Заголовок: {news.title}")
    if news.published_at:
        lines.append(f"Дата: {news.published_at}")
    lines.append(f"Суммари: {news.summary}")
    lines.append(f"Текст:\n{news.text}")
    return "\n".join(lines)


def _passages_block(passages: list[BankPassage]) -> str:
    if not passages:
        return "=== АНАЛИТИКА АЛЬФА-БАНКА ===\n(релевантных материалов нет)"
    lines = ["=== АНАЛИТИКА АЛЬФА-БАНКА ==="]
    for number, passage in enumerate(passages, start=1):
        ref = f"bank:{passage.document_id}"
        if passage.chunk_id:
            ref += f"#{passage.chunk_id}"
        published = f", {passage.published_at}" if passage.published_at else ""
        lines.append(f"[B{number}] {ref} — {passage.title}{published}\n{passage.text}")
    return "\n\n".join(lines)


def _draft_block(draft: InsightDraft) -> str:
    return "=== ЧЕРНОВИК ИНСАЙТА ===\n" + draft.model_dump_json(indent=2)


def _verification_block(verification: Verification) -> str:
    return "=== ЗАМЕЧАНИЯ ПРОВЕРЯЮЩЕГО ===\n" + verification.model_dump_json(indent=2)
