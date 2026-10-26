import difflib
import json
import logging
import re
import time

from app.schemas.chat import ChatAnswerPayload, ChatSource
from app.schemas.company_news import InterpretationSource
from app.services.company_news import DeadlineClient, require_llm
from app.services.llm.config import ServicesSettings
from app.services.llm.insight import answer_question_structured
from app.services.llm.provider import MockProvider, OpenAICompatProvider
from app.services.llm.schemas import BankPassage, CompanyContext, InsightDraft, NewsEvent
from app.services.llm.text import strip_passage_refs

logger = logging.getLogger(__name__)

# Служебные метки источников вида [news:https://…], [company:Выручка], [bank:<id>#1] —
# источники отдаются отдельным списком, в тексте ответа они только мешают.
_INLINE_REF = re.compile(
    r"\s*(?:\[(?:news|bank|company|analytics):[^\]]*\]"
    # в круглых скобках допускается один уровень вложенности и вводное слово перед ссылкой:
    # (company:Отрасль (ОКВЭД 25.11)), (источник: news:https://…), (см. bank:…#p1)
    r"|\((?:[^()]|\([^()]*\)){0,25}?(?<!\w)(?:news|bank|company|analytics):"
    r"(?:[^()]|\([^()]*\))*\))"
)


def strip_inline_refs(text: str) -> str:
    cleaned = strip_passage_refs(_INLINE_REF.sub("", text)) or ""
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip()


MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12_000


class InvalidChatAnswer(ValueError):
    pass


def bounded_history(messages, *, exclude_message_id=None) -> str:
    """Newest complete exchanges within both message and character budgets."""

    lines: list[str] = []
    used = 0
    candidates = [
        message
        for message in messages
        if message.id != exclude_message_id
        and message.status == "ready"
        and message.content.strip()
        and message.role in {"user", "assistant"}
    ][-MAX_HISTORY_MESSAGES:]
    for message in reversed(candidates):
        label = "Пользователь" if message.role == "user" else "Ассистент"
        line = f"{label}: {message.content.strip()}"
        if used + len(line) > MAX_HISTORY_CHARS:
            remaining = MAX_HISTORY_CHARS - used
            if remaining > len(label) + 3:
                lines.append(line[:remaining])
            break
        lines.append(line)
        used += len(line)
    return "\n".join(reversed(lines))


def _source_map(
    context: CompanyContext,
    news: NewsEvent,
    sources: list[InterpretationSource],
) -> dict[str, ChatSource]:
    mapped: dict[str, ChatSource] = {}
    for source in sources:
        if source.evidence_ref.startswith("bank:"):
            kind = "bank"
        elif source.evidence_ref.startswith("company:"):
            kind = "company"
        else:
            kind = "news"
        mapped[source.evidence_ref] = ChatSource(
            evidence_ref=source.evidence_ref,
            kind=kind,
            title=source.title,
            source_url=source.source_url,
            published_at=source.published_at,
            page_number=source.page_number,
        )
    news_ref = f"news:{news.url}"
    if news.url and news_ref not in mapped:
        mapped[news_ref] = ChatSource(
            evidence_ref=news_ref,
            kind="news",
            title=news.title or "Новость",
            source_url=news.url,
            published_at=news.published_at,
        )
    for label in context.evidence_labels:
        ref = f"company:{label}"
        mapped[ref] = ChatSource(
            evidence_ref=ref,
            kind="company",
            title=f"Поле профиля компании: {label}",
        )
    return mapped


def _resolve_ref(ref: str, allowed: dict[str, ChatSource]) -> str | None:
    """Точная ссылка из списка допустимых; опечатки и пропущенный префикс исправляются."""

    cleaned = ref.strip()
    if cleaned in allowed:
        return cleaned
    for prefix in ("news:", "bank:", "company:"):
        if not cleaned.startswith(prefix) and f"{prefix}{cleaned}" in allowed:
            return f"{prefix}{cleaned}"
    if cleaned.startswith("company:"):
        # Приблизительно исправляем только метки полей профиля; URL и id документов — точно.
        labels = [ref for ref in allowed if ref.startswith("company:")]
        close = difflib.get_close_matches(cleaned, labels, n=1, cutoff=0.85)
        return close[0] if close else None
    return None


def _validate_payload(
    payload: ChatAnswerPayload, allowed: dict[str, ChatSource]
) -> tuple[str, list[ChatSource], list[str]]:
    """Проверка тезисов ответа без отказа: слабые тезисы понижаются до ai_inference.

    Раньше любая неточная ссылка давала 502 и чат не отвечал вовсе. Теперь выдуманные
    ссылки отбрасываются, факт без документа и позиция банка без bank-источника
    становятся ai_inference, а пользователь получает предупреждение.
    """

    warnings: list[str] = []
    cited: list[str] = []
    claim_kinds: set[str] = set()
    for claim in payload.claims:
        refs: list[str] = []
        for ref in claim.evidence_refs:
            resolved = _resolve_ref(ref, allowed)
            if resolved is None:
                logger.warning("Chat answer cited unknown evidence ref: %r", ref)
                warnings.append("Часть ссылок на источники не подтверждена и отброшена.")
            elif resolved not in refs:
                refs.append(resolved)
        if not refs:
            logger.warning("Chat claim dropped, no valid evidence: %r", claim.text[:120])
            continue
        kind = claim.kind
        if kind == "bank_position" and not any(ref.startswith("bank:") for ref in refs):
            kind = "ai_inference"
            warnings.append("Позиция банка без bank-источника показана как вывод AI.")
        elif kind == "fact" and not any(ref.startswith(("news:", "bank:")) for ref in refs):
            kind = "ai_inference"
            warnings.append("Факт без документа-источника показан как вывод AI.")
        claim_kinds.add(kind)
        cited.extend(refs)
    if not claim_kinds and not payload.missing_evidence:
        logger.warning("Chat answer has neither valid claims nor missing_evidence")
        warnings.append("Ответ не подкреплён проверяемыми источниками.")
    answer_kind = next(iter(claim_kinds)) if len(claim_kinds) == 1 else "ai_inference"
    unique_refs = list(dict.fromkeys(cited))
    return answer_kind, [allowed[ref] for ref in unique_refs], list(dict.fromkeys(warnings))


def generate_chat_answer(
    context: CompanyContext,
    news: NewsEvent,
    insight: InsightDraft,
    passages: list[BankPassage],
    sources: list[InterpretationSource],
    question: str,
    history: str,
    settings: ServicesSettings,
) -> tuple[str, str, list[dict], list[str]]:
    require_llm(settings)
    allowed = _source_map(context, news, sources)
    refs = json.dumps(list(allowed), ensure_ascii=False)
    augmented_question = (
        question.strip()
        + "\n\nВерни JSON: {answer: str, claims: [{text: str, kind: "
        + '"fact|bank_position|ai_inference", evidence_refs: [str]}], '
        + "missing_evidence: [str]}. Каждый существенный тезис вынеси в claims. "
        + f"evidence_refs должны точно совпадать с одним из: {refs}. "
        + "bank_position допустим только для дословной позиции банка из bank-источника; "
        + "персональный вывод на основе bank-источника остаётся ai_inference. "
        + "Если подтверждения нет, не создавай claim и перечисли пробел в missing_evidence. "
        + "Не пытайся восстановить реальные имена или реквизиты из обезличенных данных."
    )
    if settings.llm_mock:
        provider = MockProvider(
            handler=lambda purpose, messages, json_mode: json.dumps(
                {
                    "answer": "Данных недостаточно для подтверждённого ответа.",
                    "claims": [],
                    "missing_evidence": ["В демонстрационном режиме нет ответа модели"],
                },
                ensure_ascii=False,
            )
        )
        payload, _ = answer_question_structured(
            context,
            news,
            insight,
            augmented_question,
            ChatAnswerPayload,
            passages,
            history=history,
            llm=provider,
            settings=settings,
        )
    else:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        with DeadlineClient(
            deadline=time.monotonic() + settings.interpretation_timeout_seconds,
            per_call_timeout=settings.llm_timeout_seconds,
            base_url=settings.llm_api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"} if key else {},
        ) as client:
            provider = OpenAICompatProvider(
                settings.llm_api_base,
                key,
                settings.llm_timeout_seconds,
                client=client,
                reasoning_budget=settings.llm_reasoning_budget,
            )
            payload, _ = answer_question_structured(
                context,
                news,
                insight,
                augmented_question,
                ChatAnswerPayload,
                passages,
                history=history,
                llm=provider,
                settings=settings,
            )
    answer = strip_inline_refs(payload.answer) or payload.answer.strip()
    answer_kind, cited, warnings = _validate_payload(payload, allowed)
    if payload.missing_evidence:
        warnings.extend(f"Недостаточно данных: {item}" for item in payload.missing_evidence)
    if settings.llm_mock:
        warnings.append("Демонстрационный режим: ответ не является аналитикой LLM.")
    return answer, answer_kind, [source.model_dump(mode="json") for source in cited], warnings
