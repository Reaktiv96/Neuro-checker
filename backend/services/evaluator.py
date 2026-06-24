"""
LLM evaluation service.

Evaluates student code against a reference assignment using OpenAI,
incorporating deterministic precheck results (forced_score if applicable).
"""
import json
import logging
import ast
import re
from typing import Optional
from openai import OpenAI

from backend.config import Config
from backend.models import EvaluationResult, PrecheckResult
from backend.models.prompts import PROMPTS
from backend.services.precheck import run_precheck
from backend.services.code_cleaner import clean_notebook_code

logger = logging.getLogger(__name__)


def _extract_semantic_hints(code: str) -> str:
    """Build deterministic semantic hints to help LLM avoid missing critical logic errors."""
    hints: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Нет (код не парсится для статического анализа)."

    assigned: dict[str, int] = {}
    used: set[str] = set()

    class Analyzer(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = assigned.get(target.id, 0) + 1
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assigned[node.target.id] = assigned.get(node.target.id, 0) + 1
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                used.add(node.id)
            self.generic_visit(node)

    Analyzer().visit(tree)

    unused = [name for name in assigned if name not in used and not name.startswith('_')]
    if unused:
        hints.append(
            "Обнаружены присвоенные, но неиспользуемые переменные: " + ", ".join(sorted(unused)[:12])
        )

    critical_like = [
        name for name in unused
        if re.search(r'kb|knowledge|context|retriev|rag|base|info', name, re.IGNORECASE)
    ]
    if critical_like:
        hints.append(
            "ВНИМАНИЕ: среди неиспользуемых переменных есть потенциально критичные для требований контекста/базы знаний: "
            + ", ".join(sorted(critical_like))
        )

    return "\n".join(f"- {h}" for h in hints) if hints else "Нет явных семантических подсказок."


def _extract_semantic_analysis(code: str) -> dict:
    """Return structured semantic analysis used both for prompting and post-validation."""
    hints: list[str] = []
    has_critical_issue = False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "hints_text": "Нет (код не парсится для статического анализа).",
            "has_critical_issue": True,
            "unused_variables": [],
            "critical_unused_variables": [],
        }

    assigned: dict[str, int] = {}
    used: set[str] = set()

    class Analyzer(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = assigned.get(target.id, 0) + 1
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assigned[node.target.id] = assigned.get(node.target.id, 0) + 1
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                used.add(node.id)
            self.generic_visit(node)

    Analyzer().visit(tree)

    unused = [name for name in assigned if name not in used and not name.startswith('_')]
    if unused:
        hints.append(
            "Обнаружены присвоенные, но неиспользуемые переменные: " + ", ".join(sorted(unused)[:12])
        )

    # Detect unused variables assigned from any function call — regardless of variable name.
    # This catches data-flow breaks like: docs = retrieve(...) / sim_docs = db.search(...) / result = api()
    # when the variable is never passed to the LLM template or stored as required.
    call_assigned: set[str] = set()

    def _has_call(node: ast.expr) -> bool:
        return any(isinstance(n, ast.Call) for n in ast.walk(node))

    class CallAssignFinder(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            if _has_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        call_assigned.add(target.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if node.value is not None and _has_call(node.value):
                if isinstance(node.target, ast.Name):
                    call_assigned.add(node.target.id)
            self.generic_visit(node)

    CallAssignFinder().visit(tree)

    critical_call_unused = [name for name in unused if name in call_assigned]
    if critical_call_unused:
        has_critical_issue = True
        hints.append(
            "КРИТИЧНО: следующие переменные получены вызовом функции, но нигде не используются "
            "— возможен обрыв потока данных (результат получен, но не передан туда, где требуется): "
            + ", ".join(sorted(critical_call_unused))
            + ". Проверь, что каждая из них действительно доходит до финального места использования "
            "(шаблон/аргументы LLM-вызова/хранилище) — если нет, требование фактически не выполнено."
        )

    # Detect variables used but never assigned inside function bodies (potential NameErrors).
    _BUILTINS = frozenset(dir(__builtins__)) if isinstance(__builtins__, dict) else frozenset(dir(__builtins__))
    undefined_in_funcs: list[str] = []
    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_scope: set[str] = set()
        func_used_names: set[str] = set()

        # Collect parameter names as part of local scope
        all_args = (
            func_node.args.args
            + func_node.args.posonlyargs
            + func_node.args.kwonlyargs
        )
        for arg in all_args:
            func_scope.add(arg.arg)
        if func_node.args.vararg:
            func_scope.add(func_node.args.vararg.arg)
        if func_node.args.kwarg:
            func_scope.add(func_node.args.kwarg.arg)

        # Walk function body for assignments and uses
        for child in ast.walk(func_node):
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        func_scope.add(t.id)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
                target = child.target if hasattr(child, 'target') else None
                if target and isinstance(target, ast.Name):
                    func_scope.add(target.id)
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    func_scope.add(child.target.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                func_used_names.add(child.id)

        # Names referenced but not locally defined and not globals/builtins
        potentially_undefined = [
            name for name in func_used_names
            if name not in func_scope
            and name not in assigned          # not a module-level variable
            and name not in _BUILTINS
            and name not in used              # not used at module level (i.e. not a global ref)
            and not name[0].isupper()         # skip CONSTANTS
            and not name.startswith('_')
            and name not in ('self', 'cls', 'True', 'False', 'None')
        ]
        undefined_in_funcs.extend(potentially_undefined)

    if undefined_in_funcs:
        has_critical_issue = True
        unique_undef = sorted(set(undefined_in_funcs))
        hints.append(
            "КРИТИЧНО: в теле функции используются переменные, которые нигде не присвоены (возможен NameError при вызове): "
            + ", ".join(unique_undef[:8])
        )

    # Detect function parameters related to knowledge base that are never used in the
    # function body — catches logical errors where knowledge_base is bypassed, e.g.
    # generate_rag_prompt(user_question, '') instead of generate_rag_prompt(user_question, knowledge_base).
    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_params = [
            arg.arg for arg in
            func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs
        ]
        kb_params = [
            p for p in all_params
            if re.search(r'knowledge', p, re.IGNORECASE) or p == 'kb'
        ]
        if not kb_params:
            continue
        func_body_used: set[str] = set()
        for stmt in func_node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    func_body_used.add(child.id)
        unused_kb_params = [p for p in kb_params if p not in func_body_used]
        if unused_kb_params:
            has_critical_issue = True
            hints.append(
                f"КРИТИЧНО: параметр(ы), связанные с базой знаний, переданы в функцию "
                f"`{func_node.name}` но нигде не используются в её теле "
                "(вероятно, база знаний игнорируется и не передаётся в промпт): "
                + ", ".join(sorted(unused_kb_params))
            )

    # Detect broken RAG pipelines (search not in messages / index never queried).
    for rag_issue in _check_all_rag_issues(tree):
        has_critical_issue = True
        hints.append(rag_issue)

    return {
        "hints_text": "\n".join(f"- {h}" for h in hints) if hints else "Нет явных семантических подсказок.",
        "has_critical_issue": has_critical_issue,
        "unused_variables": sorted(unused),
        "critical_unused_variables": sorted(critical_call_unused),
        "undefined_in_functions": sorted(set(undefined_in_funcs)),
    }


from backend.utils.rag_checks import check_all_rag_issues as _check_all_rag_issues


def _is_kb_info_used_in_prompt(code: str) -> bool:
    """Detect typical pattern where kb_info is actually included in system_prompt sent to LLM."""
    has_kb_var = re.search(r'\bkb_info\b', code) is not None
    has_kb_in_prompt = re.search(
        r'system_prompt\s*=\s*f?[\'\"]{3}[\s\S]*\{\s*kb_info\s*\}[\s\S]*[\'\"]{3}',
        code,
    ) is not None
    sends_system_prompt = re.search(r'[\'\"]content[\'\"]\s*:\s*system_prompt', code) is not None
    return has_kb_var and has_kb_in_prompt and sends_system_prompt


def _is_speculative_penalty(comment: str) -> bool:
    text = (comment or '').lower()
    speculative_patterns = [
        r'может\s+быть',
        r'может\s+не',
        r'может\s+ограничи',     # "может ограничить / ограничивать"
        r'возможно',
        r'рекомендуется',
        r'можно\s+улучшить',
        r'стоит\s+добав',
        r'стоит\s+рассмотр',     # "стоит рассмотреть"
        r'в\s+будущем',          # "в будущем стоит / можно"
        r'в\s+дальнейшем',       # "в дальнейшем можно"
        r'более\s+сложн',
    ]
    return any(re.search(p, text) for p in speculative_patterns)


def _has_contradiction_with_code(comment: str, code: str) -> bool:
    """Detect known false-positive claims that contradict the actual code."""
    text = (comment or '').lower()

    # False claim: kb_info not passed to prompt, while it is actually passed.
    kb_claim = (
        ('kb_info' in text or 'баз' in text or 'knowledge' in text)
        and ('не перед' in text or 'исключ' in text or 'не использ' in text)
    )
    if kb_claim and _is_kb_info_used_in_prompt(code):
        return True

    # False claim: FSM states are unused while decorators reference them.
    fsm_unused_claim = ('fsm' in text or 'состояни' in text) and ('не использ' in text)
    fsm_used_in_decorators = re.search(r'@dp\.message\(UserProfile\.', code) is not None
    if fsm_unused_claim and fsm_used_in_decorators:
        return True

    return False


def _filter_unsupported_fragments(text: str, code: str) -> str:
    """Remove speculative/contradicting fragments from visible feedback text."""
    raw = (text or '').replace('[Пост-валидатор]', '').strip()
    if not raw:
        return ''

    # Split by lines first; if long paragraph, additionally split by sentence boundary.
    chunks: list[str] = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        sub = [s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if s.strip()]
        chunks.extend(sub if sub else [line])

    kept = []
    for chunk in chunks:
        if _is_speculative_penalty(chunk):
            continue
        if _has_contradiction_with_code(chunk, code):
            continue
        kept.append(chunk)

    return '\n'.join(kept).strip()


def _filter_perfect_score_noise(text: str) -> str:
    """Remove downgrade-like boilerplate that conflicts with a 10/10 result."""
    raw = (text or '').replace('[Пост-валидатор]', '').strip()
    if not raw:
        return ''

    noise_patterns = [
        r'близк\w*\s+к\s+идеал',
        r'требу\w*\s+небольш\w*\s+доработ',
        r'для\s+полного\s+соответств',
        r'обратите\s+внимани',
        r'рекомендаци',
        r'это\s+улучшит\s+функциональност',
    ]

    chunks: list[str] = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        sub = [s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if s.strip()]
        chunks.extend(sub if sub else [line])

    kept: list[str] = []
    for chunk in chunks:
        low = chunk.lower()
        if any(re.search(p, low) for p in noise_patterns):
            continue
        kept.append(chunk)

    return '\n'.join(kept).strip()


def _apply_post_validator(content: dict, cleaned_code: str, precheck_dict: dict, semantic_analysis: dict) -> dict:
    """Evidence-based post validation: rollback unsupported score reductions."""
    # If deterministic precheck already forced the score, respect it.
    if precheck_dict.get('forced_score') is not None:
        return content

    # If there is an explicit critical semantic signal, do not relax score automatically.
    if semantic_analysis.get('has_critical_issue'):
        return content

    tasks = content.get('homework_tasks') or []
    unsupported_reasons = []

    for task in tasks:
        score = task.get('score')
        comment = task.get('comment') or ''
        cleaned_comment = _filter_unsupported_fragments(comment, cleaned_code)
        if comment != cleaned_comment:
            task['comment'] = cleaned_comment

        if isinstance(score, (int, float)) and score < 10:
            if _is_speculative_penalty(comment) or _has_contradiction_with_code(comment, cleaned_code):
                unsupported_reasons.append(comment)
                task['score'] = 10
                task['comment'] = cleaned_comment or "Решение соответствует требованиям задания. Существенных подтвержденных ошибок не обнаружено."

    # Fallback for outputs where score was reduced only in overall_comment.
    overall_score = content.get('overall_score')
    overall_comment = content.get('overall_comment') or ''
    filtered_overall = _filter_unsupported_fragments(overall_comment, cleaned_code)
    if overall_comment != filtered_overall:
        content['overall_comment'] = filtered_overall or "Решение соответствует требованиям задания."

    if (not tasks) and isinstance(overall_score, (int, float)) and overall_score < 10:
        if _is_speculative_penalty(overall_comment) or _has_contradiction_with_code(overall_comment, cleaned_code):
            unsupported_reasons.append(overall_comment)
            content['overall_score'] = 10
            content['overall_comment'] = filtered_overall or "Решение соответствует требованиям задания. Существенных подтвержденных ошибок не обнаружено."

    additional_recommendations = content.get('additional_recommendations') or ''
    filtered_reco = _filter_unsupported_fragments(additional_recommendations, cleaned_code)
    if additional_recommendations != filtered_reco:
        content['additional_recommendations'] = filtered_reco

    # Recompute overall score from tasks if we changed any per-task score.
    if unsupported_reasons and tasks:
        task_scores = [t.get('score') for t in tasks if isinstance(t.get('score'), (int, float))]
        if task_scores:
            content['overall_score'] = int(round(sum(task_scores) / len(task_scores)))
        if not (content.get('overall_comment') or '').strip():
            content['overall_comment'] = "Решение соответствует требованиям задания."

    # Keep wording consistent with perfect score.
    final_score = content.get('overall_score')
    if not isinstance(final_score, (int, float)) and tasks:
        task_scores = [t.get('score') for t in tasks if isinstance(t.get('score'), (int, float))]
        if task_scores:
            final_score = int(round(sum(task_scores) / len(task_scores)))
            content['overall_score'] = final_score

    if isinstance(final_score, (int, float)) and final_score >= 10:
        content['overall_comment'] = (
            _filter_perfect_score_noise(content.get('overall_comment') or '')
            or "Решение полностью соответствует требованиям задания."
        )
        content['additional_recommendations'] = _filter_perfect_score_noise(
            content.get('additional_recommendations') or ''
        )
        for task in tasks:
            task['comment'] = _filter_perfect_score_noise(task.get('comment') or '') or task.get('comment') or ''

    content['post_validation'] = {
        'unsupported_reasons_count': len(unsupported_reasons),
        'score_adjusted': bool(unsupported_reasons),
    }

    return content


def _get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or Config.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def evaluate_homework(
    tema: str,
    zadanie: str,
    resheniye: str,
    name: str = "Участник",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    execution_logs: Optional[str] = None,
    enable_post_validator: bool = True,
    assignment_type: str = "code",
) -> dict:
    """
    Evaluate student homework against an assignment.

    Pipeline:
    1. Run deterministic precheck on raw submission
    2. Clean code
    3. Build prompts (inject precheck data)
    4. Call LLM
    5. Enforce forced_score from precheck if applicable

    Args:
        tema: Assignment topic
        zadanie: Assignment text / goal
        resheniye: Raw student submission
        name: Student name for personalised feedback
        model: OpenAI model name
        api_key: Optional user-supplied API key

    Returns:
        Dict with full evaluation result + usage stats
    """
    client = _get_openai_client(api_key)

    # Text-only path: explicit assignment_type='text' flag (Option A).
    # Fallback heuristic kept for calls that don't pass the flag yet:
    # no code but substantial text in execution_logs.
    is_text_only = assignment_type == "text" or (
        assignment_type == "code"
        and (not resheniye or len(resheniye.strip()) < 10)
        and bool(execution_logs and len(execution_logs.strip()) > 100)
    )
    if is_text_only:
        system_prompt = PROMPTS["evaluation_system_text"].replace("{name}", name)
        user_prompt = (
            PROMPTS["evaluation_user_text"]
            .replace("{dz}", zadanie)
            .replace("{resheniye}", execution_logs.strip())
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content or "{}"
        content = json.loads(raw_content)
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        content["precheck"] = {
            "has_valid_code": False,
            "has_syntax_errors": False,
            "has_stubs": False,
            "has_metadata": False,
            "forced_score": None,
            "reasons": [],
        }
        content["usage"] = usage
        logger.info("Text-only evaluation path used (no code, logs used as submission)")
        return content

    # Step 1: Deterministic precheck
    precheck = run_precheck(resheniye)
    precheck_dict = {
        "has_valid_code": precheck.has_valid_code,
        "has_syntax_errors": precheck.has_syntax_errors,
        "has_stubs": precheck.has_stubs,
        "has_metadata": precheck.has_metadata,
        "forced_score": precheck.forced_score,
        "reasons": precheck.reasons,
    }

    # Step 2: Clean code for LLM
    cleaned = clean_notebook_code(resheniye)
    semantic_analysis = _extract_semantic_analysis(cleaned)
    semantic_hints = semantic_analysis['hints_text']

    # Step 3: Build prompts
    system_prompt = PROMPTS["evaluation_system"].replace("{name}", name)

    user_prompt = (
        PROMPTS["evaluation_user"]
        .replace("{dz}", zadanie)
        .replace("{resheniye}", cleaned)
        .replace("{precheck_json}", json.dumps(precheck_dict, ensure_ascii=False, indent=2))
        .replace("{execution_logs}", execution_logs.strip() if execution_logs and execution_logs.strip() else "Нет дополнительных результатов выполнения")
        .replace("{semantic_hints}", semantic_hints)
    )

    # Step 4: Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw_content = response.choices[0].message.content or "{}"
    content = json.loads(raw_content)

    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }

    # Step 5: Enforce forced_score
    if precheck.forced_score is not None:
        ai_score = content.get("overall_score")
        logger.info(
            f"Enforcing forced_score={precheck.forced_score} "
            f"over AI score={ai_score}"
        )
        content["overall_score"] = precheck.forced_score
        if precheck.reasons:
            precheck_note = " ".join(precheck.reasons)
            existing = content.get("overall_comment", "")
            content["overall_comment"] = (
                f"[АВТОМАТИЧЕСКАЯ ПРОВЕРКА] {precheck_note}\n\n{existing}"
            ).strip()

    # Step 6: Evidence-based post validation (disabled for synthetic incorrect cases).
    if enable_post_validator:
        content = _apply_post_validator(content, cleaned, precheck_dict, semantic_analysis)

    content["precheck"] = precheck_dict
    content["usage"] = usage

    return content
