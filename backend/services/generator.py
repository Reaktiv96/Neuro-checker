"""
Case generation service.

Generates correct alternative solutions and incorrect cases (with various error types)
based on a reference (etalon) Python code.

Core flow:
1. Send etalon code to LLM with a generation prompt
2. LLM returns a PATCH (find/replace pair), not full code
3. Apply patch to etalon code
4. Validate syntax (and for syntax_error type - verify syntax IS broken)
5. Retry up to MAX_ATTEMPTS on failure
"""
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from openai import OpenAI

from backend.config import Config
from backend.models import GeneratedCase
from backend.models.prompts import PROMPTS, INCORRECT_CASE_TYPES
from backend.utils.syntax_validator import validate_python_syntax
from backend.utils.patch_applier import apply_patches

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

# Pricing per 1M tokens (input, output) in USD
# Source: OpenAI pricing page
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":          (2.50,  10.00),
    "gpt-4o-mini":     (0.15,   0.60),
    "gpt-4-turbo":     (10.00, 30.00),
    "gpt-4":           (30.00, 60.00),
    "gpt-3.5-turbo":   (0.50,   1.50),
    "gpt-4.1":         (2.00,   8.00),
    "gpt-4.1-mini":    (0.40,   1.60),
    "gpt-4.1-nano":    (0.10,   0.40),
    "o1":              (15.00, 60.00),
    "o1-mini":         (3.00,  12.00),
    "o3-mini":         (1.10,   4.40),
    "o4-mini":         (1.10,   4.40),
}


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost for a given model and token counts."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Prefix match (e.g. "gpt-4o-2024-11-20" → "gpt-4o")
        for key, val in MODEL_PRICING.items():
            if model.startswith(key):
                pricing = val
                break
    if pricing is None:
        return 0.0
    input_per_m, output_per_m = pricing
    return (prompt_tokens * input_per_m + completion_tokens * output_per_m) / 1_000_000


def _convert_usd_to_rub(cost_usd: float) -> float:
    """Convert USD cost to RUB using configured exchange rate."""
    rate = Config.RUBLES_IN_DOLLAR
    if rate <= 0:
        return 0.0
    return cost_usd * rate


def _trace(trace_log: Optional[list], event: str, **payload) -> None:
    """Append trace event for generation debug logs."""
    if trace_log is None:
        return
    item = {
        "ts": datetime.now().isoformat(),
        "event": event,
    }
    item.update(payload)
    trace_log.append(item)


def _get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or Config.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# Extra deliverables detection & generation
# ---------------------------------------------------------------------------

_EXTRA_CONTENT_RE = re.compile(
    r'(?:тест|лог\s+работ|вывод|продемонстрир|провед[её]н|предоставить|отч[её]т|демонстрац)',
    re.IGNORECASE,
)

_CASE_TYPE_BEHAVIOR: dict[str, str] = {
    "perfect_etalon":      "эталонное рабочее решение — всё работает корректно",
    "perfect_alternative": "правильное решение в другом стиле — всё работает корректно",
    "non_optimal":         "решение работает правильно, но реализовано неоптимально (избыточные переменные/шаги)",
    "syntax_error":        "код содержит синтаксическую ошибку — при запуске возникает traceback",
    "logical_error":       "код запускается, но выдаёт неверные результаты (неправильная логика)",
    "partial":             "код запускается, но ключевая функция заменена заглушкой (pass) — бот не отвечает или падает",
    "cheating":            "код запускается, но вместо реального LLM-ответа всегда возвращается захардкоженная строка",
}


def _should_generate_extra_content(zadanie: str) -> bool:
    """Returns True if the zadanie requires deliverables beyond just code."""
    return bool(_EXTRA_CONTENT_RE.search(zadanie))


def _generate_case_extra_content(
    api_key: Optional[str],
    model: str,
    tema: str,
    zadanie: str,
    case_code: str,
    case_type: str,
    trace_log: Optional[list] = None,
) -> tuple[str, dict]:
    """
    Generate synthetic execution output / logs / conclusions for a generated case.
    This text simulates what would appear in the student's Colab notebook
    (execution output, markdown cells) beyond the code itself.
    """
    client = _get_openai_client(api_key)
    behavior = _CASE_TYPE_BEHAVIOR.get(case_type, "решение неизвестного типа")

    system = (
        "Ты помогаешь создать синтетические данные вывода Google Colab-ноутбука студента. "
        "Студент должен был предоставить несколько материалов помимо кода. "
        "Твоя задача — написать реалистичный текст, который появился бы в ячейках вывода "
        "и markdown-ячейках ноутбука. Пиши только сам текст, без JSON-обёрток."
    )

    user = (
        f"Тема: {tema}\n"
        f"Задание:\n{zadanie}\n\n"
        f"Характер варианта: {behavior}\n\n"
        f"Код варианта:\n{case_code[:3000]}\n\n"
        "Напиши реалистичные данные вывода Colab-ноутбука для этого студента согласно требованиям задания. "
        "Включи то, что требует задание (тесты, логи, выводы), НО соответствующее характеру варианта. "
        "Если код работает — логи показывают работу бота; если ошибка — traceback. "
        + (
            "ВАЖНО: этот вариант РАБОЧИЙ — все функции выполняются без ошибок, "
            "без исключений, без traceback. НЕ добавляй ошибок парсинга, "
            "AttributeError, TimeoutError или других исключений ни для одной из функций. "
            if case_type in ("perfect_alternative", "non_optimal") else ""
        )
        + "Объём: 150–350 слов. Только текст без JSON."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=600,
    )

    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    _trace(trace_log, "extra_content_generated", case_type=case_type, length=len(text))
    return text, usage


def _call_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> dict:
    """
    Call OpenAI and return parsed JSON response + usage stats.

    Returns:
        Dict with keys: data (parsed dict), usage (token counts)
    """
    import json

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return {"data": data, "usage": usage, "raw": raw}


def _detect_unused_variables_heuristic(code: str) -> list[str]:
    """
    Fast heuristic: find variables assigned with `var = ...` but never referenced again.
    Look for common patterns like `var = func(...)` followed by code that doesn't use var.
    Returns list of suspect variable names.
    """
    lines = code.split('\n')
    unused_suspects = []
    
    # Find simple assignments: `var_name = ...`
    for i, line in enumerate(lines):
        # Match patterns like "    var_name = something"
        match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*', line)
        if match:
            var_name = match.group(1)
            # Skip obvious loop/control variables
            if var_name in ['i', 'j', 'k', 'x', 'y', 'e']:
                continue
            # Check if this variable is used in the rest of the code
            rest_of_code = '\n'.join(lines[i+1:])
            if not re.search(rf'\b{re.escape(var_name)}\b', rest_of_code):
                unused_suspects.append(var_name)
    
    return unused_suspects


def _check_renamed_functions(etalon_code: str, generated_code: str) -> list[str]:
    """
    Detect incomplete renames: function was removed from definitions in generated_code
    but its old name still appears as a call. Indicates the patch renamed the def
    but forgot to update call sites.
    Returns list of broken function names.
    """
    def_pattern = re.compile(r'(?:async\s+)?def\s+(\w+)\s*\(')
    etalon_funcs = set(def_pattern.findall(etalon_code))
    generated_funcs = set(def_pattern.findall(generated_code))
    removed = etalon_funcs - generated_funcs
    broken = []
    for name in removed:
        if re.search(r'\b' + re.escape(name) + r'\s*\(', generated_code):
            broken.append(name)
    return broken


def _validate_logical_error_case(
    client: OpenAI,
    model: str,
    tema: str,
    zadanie: str,
    etalon_code: str,
    generated_code: str,
    change_summary: str,
) -> dict:
    """
    Verify that logical_error case truly matches its description and task requirements.

    Returns dict:
    {
      "valid": bool,
      "reason": str
    }
    """
    # ── Deterministic pre-gate: RAG pipeline check ────────────────────────────
    # If change_summary claims similarity_search results are not used in messages,
    # verify this with a static AST check before calling the LLM.
    # This catches the common failure mode where the generator writes a correct
    # RAG function (results DO flow to messages) but writes a wrong summary.
    #
    # CAPABILITY GUARD: only apply this rejection when the code actually uses a
    # vector-store RAG mechanism our static checks understand. For web-search /
    # tool agents (Tavily, SerpAPI), our checks can't trace the data flow, so
    # "no issue found" is meaningless — we must defer to the LLM validator
    # instead of falsely rejecting a valid logical_error case (which would
    # exhaust retries and fall back to the etalon, scoring 10/10).
    from backend.utils.rag_checks import (
        check_all_rag_issues,
        code_uses_vector_store_rag,
        summary_claims_rag_issue,
    )
    if summary_claims_rag_issue(change_summary):
        try:
            import ast as _ast
            tree = _ast.parse(generated_code)
            if code_uses_vector_store_rag(tree):
                rag_issues = check_all_rag_issues(tree)
                if not rag_issues:
                    return {
                        "valid": False,
                        "reason": (
                            "Детерминированная AST-проверка: change_summary утверждает, что результаты "
                            "similarity_search не включены в messages, но статический анализ кода "
                            "показывает обратное — результаты поиска фактически присутствуют в messages. "
                            "Баг не был применён генератором."
                        ),
                        "usage": {},
                    }
        except SyntaxError:
            pass  # code is syntactically broken — let LLM validate

    # ── Heuristic guard: variable mentioned in summary but unused ─────────────
    referenced_vars = _extract_backticked_identifiers(change_summary)
    unused_vars = _detect_unused_variables_heuristic(generated_code)

    suspicious = [v for v in referenced_vars if v in unused_vars]
    if suspicious:
        if any(phrase in change_summary.lower() for phrase in ['исключен', 'удален', 'не использ', 'не передаю']):
            pass  # Let LLM validate further
        elif 'переменная' in change_summary.lower() or 'вычисля' in change_summary.lower():
            pass  # Let LLM validate

    system_prompt = (
        "Ты валидатор синтетических incorrect-кейсов. "
        "Проверь соответствие change_summary фактическому коду и заданию. "
        "Особенно внимательно проверь: если summary упоминает, что переменная исключена или не используется, "
        "убедись что она действительно вычисляется но не применяется в логике. "
        "Верни JSON: {'valid': true|false, 'reason': '...'}"
    )

    user_prompt = (
        "Проверь logical_error-кейс.\n\n"
        "Критерии valid=true — ДОЛЖНЫ ВЫПОЛНЯТЬСЯ ВСЕ:\n"
        "1) Ошибка нарушает ЯВНОЕ требование задания (не предположение).\n"
        "2) Ошибка действительно присутствует в сгенерированном коде.\n"
        "3) change_summary правдиво описывает реальную ошибку.\n"
        "4) Если summary утверждает проблему (например, отсутствие ключа), "
        "а в коде этой проблемы нет — valid=false.\n"
        "5) ВАЖНО: если summary говорит, что какая-то переменная/значение исключено из использования, "
        "убедись что эта переменная действительно вычисляется но НЕ используется в ключевой логике. "
        "Если переменная вообще не вычисляется или используется нормально — valid=false.\n"
        "6) КРИТИЧЕСКИ ВАЖНО — проверь, что требование ДЕЙСТВИТЕЛЬНО не выполняется: "
        "даже если в коде есть описанная 'ошибка', требование может выполняться альтернативным способом "
        "(через другой механизм, другую переменную или другой путь исполнения). "
        "Если несмотря на описанную ошибку все требования задания всё равно выполняются в сгенерированном коде — valid=false.\n"
        "7) Для требований хранения, передачи или использования данных: проверь, что нужные данные "
        "фактически НЕ сохраняются и НЕ передаются туда, где задание требует их использовать. "
        "Если данные инициализируются/вычисляются но не сохраняются/не передаются в нужное место — это ошибка (valid=true). "
        "Если данные доходят до нужного места через любой другой механизм — valid=false.\n\n"
        f"Тема: {tema}\n"
        f"Задание: {zadanie}\n\n"
        f"Эталонный код:\n{etalon_code}\n\n"
        f"Сгенерированный код:\n{generated_code}\n\n"
        f"change_summary: {change_summary}\n"
    )

    result = _call_llm(client, model, system_prompt, user_prompt)
    data = result.get("data", {})
    return {
        "valid": bool(data.get("valid", False)),
        "reason": str(data.get("reason", "")),
        "usage": result.get("usage", {}),
    }


def _extract_backticked_identifiers(text: str) -> list[str]:
    """Extract unique identifiers mentioned as `identifier` in summary text."""
    matches = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text or "")
    seen = set()
    identifiers = []
    for item in matches:
        if item not in seen:
            seen.add(item)
            identifiers.append(item)
    return identifiers


def _validate_non_optimal_case(
    client: OpenAI,
    model: str,
    tema: str,
    zadanie: str,
    etalon_code: str,
    generated_code: str,
    change_summary: str,
) -> dict:
    """
    Verify that non_optimal case matches description and keeps functionality.

    Returns dict:
    {
      "valid": bool,
      "reason": str
    }
    """
    # Fast deterministic guard: if summary explicitly names identifiers in backticks,
    # those identifiers must exist in generated code OR in the original etalon code.
    # Identifiers that were in etalon but removed in generated are valid non_optimal changes.
    referenced = _extract_backticked_identifiers(change_summary)
    missing = [
        name for name in referenced
        if re.search(rf"\b{re.escape(name)}\b", generated_code) is None
        and re.search(rf"\b{re.escape(name)}\b", etalon_code) is None
    ]
    if missing:
        return {
            "valid": False,
            "reason": (
                "В summary упомянуты идентификаторы, которых нет ни в сгенерированном коде, ни в эталоне: "
                + ", ".join(missing)
            ),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    system_prompt = (
        "Ты валидатор synthetic non_optimal-кейсов. "
        "Проверь соответствие change_summary фактическому коду и заданию. "
        "Верни JSON: {'valid': true|false, 'reason': '...'}"
    )

    user_prompt = (
        "Проверь non_optimal-кейс.\n\n"
        "Критерии valid=true:\n"
        "1) Код в целом сохраняет функциональность относительно задания (это не logical_error и не partial).\n"
        "2) В коде действительно есть именно неэффективность/избыточность, а не выдуманная проблема.\n"
        "3) change_summary правдиво описывает фактическое изменение.\n"
        "4) Если summary утверждает наличие сущности (переменной/функции), которой нет в коде, это invalid=false.\n"
        "5) Если summary противоречит коду (например, пишет про лишнюю переменную, а ее нет), это invalid=false.\n\n"
        f"Тема: {tema}\n"
        f"Задание: {zadanie}\n\n"
        f"Эталонный код:\n{etalon_code}\n\n"
        f"Сгенерированный код:\n{generated_code}\n\n"
        f"change_summary: {change_summary}\n"
    )

    result = _call_llm(client, model, system_prompt, user_prompt)
    data = result.get("data", {})
    return {
        "valid": bool(data.get("valid", False)),
        "reason": str(data.get("reason", "")),
        "usage": result.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_correct_alternative(
    etalon_code: str,
    tema: str,
    zadanie: str,
    index: int,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    trace_log: Optional[list] = None,
    custom_hint: Optional[str] = None,
) -> tuple[GeneratedCase, dict]:
    """
    Generate one correct alternative solution (different style, same result).

    Tries up to MAX_ATTEMPTS. On full failure falls back to returning etalon.

    Args:
        etalon_code: Clean reference Python code
        tema: Assignment topic
        zadanie: Assignment description
        index: Variant number (for uniqueness)
        model: OpenAI model name
        api_key: Optional user-supplied API key

    Returns:
        GeneratedCase with type="perfect_alternative"
    """
    client = _get_openai_client(api_key)
    prompt_tokens = 0
    completion_tokens = 0
    system_prompt = PROMPTS["generation_correct_system"]
    user_prompt = (
        f"Тема: {tema}\n"
        f"Задание: {zadanie}\n"
        f"Номер варианта: {index}\n"
        f"Эталонный код:\n{etalon_code}"
    )
    if custom_hint:
        user_prompt += f"\n\nУказание по изменению: {custom_hint}"
    last_failure_note = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _trace(trace_log, "correct_attempt_start", index=index, attempt=attempt)
            attempt_prompt = user_prompt
            if last_failure_note:
                attempt_prompt += "\n\n" + last_failure_note

            result = _call_llm(client, model, system_prompt, attempt_prompt)
            data = result["data"]
            usage = result.get("usage", {})
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)

            _trace(
                trace_log,
                "correct_llm_response",
                index=index,
                attempt=attempt,
                usage=usage,
                change_summary=data.get("change_summary", ""),
            )

            patch_result = apply_patches(etalon_code, data.get("patch", []))
            if not patch_result["success"]:
                last_failure_note = (
                    "Предыдущая попытка не применила патч. "
                    "Сформируй МИНИМАЛЬНЫЙ валидный patch с точным find/replace и без неоднозначных совпадений."
                )
                _trace(
                    trace_log,
                    "correct_patch_failed",
                    index=index,
                    attempt=attempt,
                    error=patch_result.get("error"),
                )
                logger.warning(
                    f"[correct #{index}] attempt {attempt}: patch failed — {patch_result['error']}"
                )
                continue

            syntax_check = validate_python_syntax(patch_result["result"])
            if not syntax_check["valid"]:
                last_failure_note = (
                    "Предыдущая попытка дала синтаксически невалидный код. "
                    f"Ошибка: {syntax_check.get('error', 'unknown')}. "
                    "Сделай безопасные правки и сохрани полностью валидный Python-синтаксис."
                )
                _trace(
                    trace_log,
                    "correct_syntax_invalid",
                    index=index,
                    attempt=attempt,
                    syntax_error=syntax_check.get("error"),
                )
                logger.warning(
                    f"[correct #{index}] attempt {attempt}: syntax error in generated code"
                )
                continue

            broken_renames = _check_renamed_functions(etalon_code, patch_result["result"])
            if broken_renames:
                names_str = ", ".join(f"`{n}`" for n in broken_renames)
                last_failure_note = (
                    f"Предыдущая попытка переименовала функцию(и) {names_str}, "
                    "но не обновила ВСЕ места вызова в файле. "
                    "Если переименовываешь функцию, patch ОБЯЗАН содержать замену "
                    "во ВСЕХ местах её вызова по всему коду."
                )
                _trace(
                    trace_log,
                    "correct_broken_rename",
                    index=index,
                    attempt=attempt,
                    broken=broken_renames,
                )
                logger.warning(
                    f"[correct #{index}] attempt {attempt}: incomplete rename — {broken_renames}"
                )
                continue

            logger.info(f"[correct #{index}] attempt {attempt}: OK")
            _trace(trace_log, "correct_success", index=index, attempt=attempt)
            return GeneratedCase(
                type="perfect_alternative",
                student_code=patch_result["result"],
                expected_grade=10,
                change_summary=data.get("change_summary", ""),
            ), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        except Exception as e:
            _trace(
                trace_log,
                "correct_exception",
                index=index,
                attempt=attempt,
                error=str(e),
            )
            logger.error(f"[correct #{index}] attempt {attempt}: exception — {e}")

    # Fallback
    logger.warning(f"[correct #{index}] all attempts failed — returning etalon")
    _trace(trace_log, "correct_fallback", index=index)
    return GeneratedCase(
        type="perfect_alternative",
        student_code=etalon_code,
        expected_grade=10,
        change_summary=f"Вариант #{index}: не удалось сгенерировать, возвращён эталон",
    ), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def generate_incorrect_case(
    etalon_code: str,
    tema: str,
    zadanie: str,
    error_type: Optional[str],
    index: int,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    trace_log: Optional[list] = None,
    override_desc: Optional[str] = None,
    prev_summaries: Optional[list] = None,
) -> tuple[GeneratedCase, dict]:
    """
    Generate one incorrect case with a specific error type.

    Args:
        etalon_code: Clean reference Python code
        tema: Assignment topic
        zadanie: Assignment description
        error_type: One of syntax_error|logical_error|non_optimal|partial|cheating
                    (None = random)
        index: Case number for logging
        model: OpenAI model name
        api_key: Optional user-supplied API key

    Returns:
        GeneratedCase with the error type embedded
    """
    client = _get_openai_client(api_key)
    prompt_tokens = 0
    completion_tokens = 0

    # Resolve error type
    type_obj = (
        next((t for t in INCORRECT_CASE_TYPES if t["type"] == error_type), None)
        or random.choice(INCORRECT_CASE_TYPES)
    )
    effective_desc = override_desc if override_desc else type_obj["desc"]

    system_prompt = PROMPTS["generation_incorrect_system"].replace(
        "{type}", type_obj["type"]
    ).replace("{desc}", effective_desc)

    base_user_prompt = (
        f"Тема: {tema}\n"
        f"Задание: {zadanie}\n"
        f"Эталонный код:\n{etalon_code}"
    )
    if prev_summaries:
        prev_list = "\n".join(f"- {s}" for s in prev_summaries)
        base_user_prompt += (
            f"\n\nУЖЕ СГЕНЕРИРОВАННЫЕ КЕЙСЫ ЭТОГО ЖЕ ТИПА "
            f"(выбери ДРУГУЮ функцию/место, отличное от перечисленных):\n{prev_list}"
        )
    last_failure_note = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            user_prompt = (
                base_user_prompt
                + "\n\n"
                + (
                    f"# Модификатор генерации: попытка {attempt}. "
                    f"Сгенерируй ошибку типа '{type_obj['type']}' другим способом, чем в предыдущих попытках. "
                    "Используй другой паттерн изменений и другой участок кода. "
                    f"Уникальный seed: {random.randint(10000, 99999)}."
                )
            )
            if last_failure_note:
                user_prompt += "\n\n" + last_failure_note

            _trace(
                trace_log,
                "incorrect_attempt_start",
                index=index,
                attempt=attempt,
                error_type=type_obj["type"],
            )
            result = _call_llm(client, model, system_prompt, user_prompt)
            data = result["data"]
            usage = result.get("usage", {})
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)

            _trace(
                trace_log,
                "incorrect_llm_response",
                index=index,
                attempt=attempt,
                error_type=type_obj["type"],
                usage=usage,
                change_summary=data.get("change_summary", ""),
            )

            patch_result = apply_patches(etalon_code, data.get("patch", []))
            if not patch_result["success"]:
                last_failure_note = (
                    "Предыдущая попытка не применила патч. "
                    "Сделай минимальный и точный patch (find должен существовать в эталонном коде)."
                )
                _trace(
                    trace_log,
                    "incorrect_patch_failed",
                    index=index,
                    attempt=attempt,
                    error_type=type_obj["type"],
                    error=patch_result.get("error"),
                )
                logger.warning(
                    f"[incorrect {type_obj['type']} #{index}] attempt {attempt}: "
                    f"patch failed — {patch_result['error']}"
                )
                continue

            syntax_check = validate_python_syntax(patch_result["result"])
            syntax_valid = syntax_check["valid"]

            if type_obj["type"] == "syntax_error":
                # Must be syntactically broken
                if syntax_valid:
                    last_failure_note = (
                        "Предыдущая попытка сохранила валидный синтаксис, а нужен именно syntax_error. "
                        "Внеси одну явную синтаксическую поломку (например, пропусти ':' в def/if), "
                        "но оставь остальной код неизменным."
                    )
                    _trace(
                        trace_log,
                        "incorrect_syntax_expected_invalid_but_valid",
                        index=index,
                        attempt=attempt,
                        error_type=type_obj["type"],
                    )
                    logger.warning(
                        f"[incorrect syntax_error #{index}] attempt {attempt}: "
                        f"generated code is syntactically valid, retrying"
                    )
                    continue
            else:
                # Must be syntactically correct
                if not syntax_valid:
                    last_failure_note = (
                        "Предыдущая попытка сломала синтаксис, а для этого типа ошибка должна быть логической/структурной при валидном Python. "
                        f"Исправь синтаксис. Ошибка: {syntax_check.get('error', 'unknown')}. "
                        "Сохрани рабочий синтаксис и внедри только нужный тип ошибки."
                    )
                    _trace(
                        trace_log,
                        "incorrect_syntax_expected_valid_but_invalid",
                        index=index,
                        attempt=attempt,
                        error_type=type_obj["type"],
                        syntax_error=syntax_check.get("error"),
                    )
                    logger.warning(
                        f"[incorrect {type_obj['type']} #{index}] attempt {attempt}: "
                        f"syntax error introduced, retrying"
                    )
                    continue

            if type_obj["type"] == "logical_error":
                validation = _validate_logical_error_case(
                    client=client,
                    model=model,
                    tema=tema,
                    zadanie=zadanie,
                    etalon_code=etalon_code,
                    generated_code=patch_result["result"],
                    change_summary=data.get("change_summary", ""),
                )
                val_usage = validation.get("usage", {})
                prompt_tokens += val_usage.get("prompt_tokens", 0)
                completion_tokens += val_usage.get("completion_tokens", 0)

                _trace(
                    trace_log,
                    "incorrect_logical_validation",
                    index=index,
                    attempt=attempt,
                    valid=validation.get("valid", False),
                    reason=validation.get("reason", ""),
                    usage=val_usage,
                )

                if not validation.get("valid", False):
                    logger.warning(
                        f"[incorrect logical_error #{index}] attempt {attempt}: "
                        f"validation failed — {validation.get('reason', 'no reason')}"
                    )
                    continue

            if type_obj["type"] == "non_optimal":
                validation = _validate_non_optimal_case(
                    client=client,
                    model=model,
                    tema=tema,
                    zadanie=zadanie,
                    etalon_code=etalon_code,
                    generated_code=patch_result["result"],
                    change_summary=data.get("change_summary", ""),
                )
                val_usage = validation.get("usage", {})
                prompt_tokens += val_usage.get("prompt_tokens", 0)
                completion_tokens += val_usage.get("completion_tokens", 0)

                _trace(
                    trace_log,
                    "incorrect_non_optimal_validation",
                    index=index,
                    attempt=attempt,
                    valid=validation.get("valid", False),
                    reason=validation.get("reason", ""),
                    usage=val_usage,
                )

                if not validation.get("valid", False):
                    logger.warning(
                        f"[incorrect non_optimal #{index}] attempt {attempt}: "
                        f"validation failed — {validation.get('reason', 'no reason')}"
                    )
                    continue

            expected_grade = 8 if type_obj["type"] == "non_optimal" else 2
            logger.info(
                f"[incorrect {type_obj['type']} #{index}] attempt {attempt}: OK"
            )
            _trace(
                trace_log,
                "incorrect_success",
                index=index,
                attempt=attempt,
                error_type=type_obj["type"],
            )
            return GeneratedCase(
                type=type_obj["type"],
                student_code=patch_result["result"],
                expected_grade=expected_grade,
                change_summary=data.get("change_summary", ""),
            ), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        except Exception as e:
            _trace(
                trace_log,
                "incorrect_exception",
                index=index,
                attempt=attempt,
                error_type=type_obj["type"],
                error=str(e),
            )
            logger.error(
                f"[incorrect {type_obj['type']} #{index}] attempt {attempt}: exception — {e}"
            )

    # Fallback
    logger.warning(
        f"[incorrect {type_obj['type']} #{index}] all attempts failed — stub fallback"
    )
    _trace(trace_log, "incorrect_fallback", index=index, error_type=type_obj["type"])
    return GeneratedCase(
        type=type_obj["type"],
        student_code="# Ошибка генерации кейса\npass",
        expected_grade=2,
        change_summary=(
            f"Не удалось сгенерировать ошибку типа {type_obj['type']} "
            f"после {MAX_ATTEMPTS} попыток"
        ),
    ), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def generate_all_cases(
    etalon_code: str,
    tema: str,
    zadanie: str,
    num_correct: int,
    num_incorrect: int,
    model: str = "gpt-4o",
    enabled_types: Optional[list] = None,
    api_key: Optional[str] = None,
    generation_id: Optional[str] = None,
) -> dict:
    """
    Generate all cases (etalon + correct alternatives + incorrect cases).

    Args:
        etalon_code: Clean reference code
        tema: Assignment topic
        zadanie: Assignment description
        num_correct: Number of correct alternatives to generate
        num_incorrect: Number of incorrect cases to generate
        model: OpenAI model name
        enabled_types: Optional list of error types in order for incorrect cases
        api_key: Optional user API key

    Returns:
        Dict with keys: results (list of GeneratedCase dicts), total_tokens (int)
    """
    results = []
    debug_log = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    _trace(
        debug_log,
        "generation_started",
        generation_id=generation_id,
        num_correct=num_correct,
        num_incorrect=num_incorrect,
        model=model,
        enabled_types=enabled_types,
    )

    needs_extra_content = _should_generate_extra_content(zadanie)
    _trace(debug_log, "extra_content_detection", needs_extra_content=needs_extra_content)

    # Always include the perfect etalon as first case
    results.append({
        "type": "perfect_etalon",
        "student_code": etalon_code,
        "expected_grade": 10,
        "change_summary": "Эталонное решение",
    })

    # Generate correct alternatives
    for i in range(1, num_correct + 1):
        case, case_usage = generate_correct_alternative(
            etalon_code=etalon_code,
            tema=tema,
            zadanie=zadanie,
            index=i,
            model=model,
            api_key=api_key,
            trace_log=debug_log,
        )
        total_prompt_tokens += case_usage["prompt_tokens"]
        total_completion_tokens += case_usage["completion_tokens"]
        results.append({
            "type": case.type,
            "student_code": case.student_code,
            "expected_grade": case.expected_grade,
            "change_summary": case.change_summary,
        })

    # Generate incorrect cases
    # If ONLY non_optimal is selected — generate all cases as non_optimal (no cap).
    # If non_optimal is mixed with other types — cap it at 1 occurrence.
    _only_non_optimal = bool(enabled_types) and all(t == "non_optimal" for t in enabled_types)
    _NON_OPTIMAL_CAP = num_incorrect if _only_non_optimal else 1
    _non_optimal_used = 0
    _all_non_opt_types = [t["type"] for t in INCORRECT_CASE_TYPES if t["type"] != "non_optimal"]
    # Pool of non-non_optimal types to use as fallback (prefer enabled_types if available)
    _fallback_pool = (
        [t for t in (enabled_types or []) if t != "non_optimal"] or _all_non_opt_types
    )

    _generated_summaries_by_type: dict[str, list[str]] = {}

    for i in range(num_incorrect):
        if _only_non_optimal:
            error_type = "non_optimal"
        elif enabled_types and i < len(enabled_types):
            error_type = enabled_types[i]
        else:
            # Beyond explicit list: cycle through non-non_optimal fallback pool
            error_type = _fallback_pool[i % len(_fallback_pool)]

        # Enforce cap: non_optimal may appear at most once (unless only non_optimal selected)
        if error_type == "non_optimal":
            if _non_optimal_used < _NON_OPTIMAL_CAP:
                _non_optimal_used += 1
            else:
                error_type = _fallback_pool[i % len(_fallback_pool)]

        case, case_usage = generate_incorrect_case(
            etalon_code=etalon_code,
            tema=tema,
            zadanie=zadanie,
            error_type=error_type,
            index=i + 1,
            model=model,
            api_key=api_key,
            trace_log=debug_log,
            prev_summaries=_generated_summaries_by_type.get(error_type, []),
        )
        _generated_summaries_by_type.setdefault(case.type, []).append(case.change_summary)
        total_prompt_tokens += case_usage["prompt_tokens"]
        total_completion_tokens += case_usage["completion_tokens"]
        results.append({
            "type": case.type,
            "student_code": case.student_code,
            "expected_grade": case.expected_grade,
            "change_summary": case.change_summary,
        })

    # Generate extra content (execution_logs) only for cases where the bot actually runs
    # correctly: perfect_alternative and non_optimal. Failing cases (syntax_error,
    # logical_error, partial, cheating) don't need synthetic logs — the evaluator sees
    # the failure directly.
    _TYPES_NEEDING_EXTRA = {"perfect_alternative", "non_optimal"}
    if needs_extra_content:
        indices_needing_extra = [
            i for i, r in enumerate(results) if r.get("type") in _TYPES_NEEDING_EXTRA
        ]

        if indices_needing_extra:
            def _generate_extra_for(idx: int):
                r = results[idx]
                extra_logs, extra_usage = _generate_case_extra_content(
                    api_key=api_key,
                    model=model,
                    tema=tema,
                    zadanie=zadanie,
                    case_code=r["student_code"],
                    case_type=r["type"],
                    trace_log=None,  # avoid thread-unsafe trace writes
                )
                return idx, extra_logs, extra_usage

            max_workers = min(8, len(indices_needing_extra))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_generate_extra_for, idx) for idx in indices_needing_extra]
                for future in as_completed(futures):
                    idx, extra_logs, extra_usage = future.result()
                    results[idx]["execution_logs"] = extra_logs
                    total_prompt_tokens += extra_usage["prompt_tokens"]
                    total_completion_tokens += extra_usage["completion_tokens"]

    total_tokens = total_prompt_tokens + total_completion_tokens
    total_cost_usd = _calculate_cost(model, total_prompt_tokens, total_completion_tokens)

    _trace(
        debug_log,
        "generation_completed",
        generation_id=generation_id,
        total_cases=len(results),
        total_tokens=total_tokens,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cost_usd=total_cost_usd,
    )

    return {
        "results": results,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost_usd,
        "debug_log": debug_log,
        "generation_id": generation_id,
    }
