"""Deterministic precheck service - validates code without LLM."""
import ast
import re
from backend.models import PrecheckResult
from backend.services.code_cleaner import clean_notebook_code
from backend.services.code_parser import parse_code
from backend.utils.syntax_validator import validate_python_syntax


def _has_stub_patterns(code: str) -> bool:
    """Detect obvious placeholder implementations without flagging valid exception handling."""
    lowered = code.lower()
    if '# todo' in lowered or 'notimplementederror' in lowered:
        return True

    # Common synthetic fallback from generator: comments + bare pass.
    if re.match(r'^(?:\s*#.*\n)*\s*pass\s*$', code or ''):
        return True

    try:
        tree = compile(
            code,
            '<precheck_stub_scan>',
            'exec',
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
    except SyntaxError:
        return False

    def strip_docstring(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant):
            if isinstance(body[0].value.value, str):
                return body[1:]
        return body

    # Flag module-level placeholder: only pass (and optional docstring).
    module_body = strip_docstring(getattr(tree, 'body', []))
    if len(module_body) == 1 and isinstance(module_body[0], ast.Pass):
        return True

    # Flag functions/classes that are empty placeholders: only pass (and optional docstring).

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = strip_docstring(node.body)
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                return True

    return False


def _has_commented_out_response_calls(code: str) -> bool:
    """
    Detect when ALL bot response calls (message.answer / message.reply) are
    commented out and no active send call remains in the code.

    Only flags when:
      - at least one `await *.answer(` or `await *.reply(` line is commented out
      - AND there is no active (uncommented) response call of any kind:
        message.answer, message.reply, bot.send_message, etc.

    Kept for cases where the function has OTHER active awaits (e.g. db.get)
    but the response sending specifically is commented out.
    """
    has_active = False
    has_commented_answer = False

    for line in code.splitlines():
        stripped = line.strip()
        is_comment = stripped.startswith('#')

        answer_pattern = bool(re.search(r'await\s+\w+\.(?:answer|reply)\s*\(', stripped))
        send_pattern = bool(re.search(r'await\s+\w+\.send_message\s*\(', stripped))

        if answer_pattern or send_pattern:
            if is_comment:
                if answer_pattern:
                    has_commented_answer = True
            else:
                has_active = True

    return has_commented_answer and not has_active


def _async_func_has_only_commented_awaits(code: str) -> bool:
    """
    Broader check: detect any async function where ALL await calls are commented
    out while at least one commented await exists.

    This catches disabled async operations regardless of the specific method name:
    message.answer, db.save, api.call, client.send, etc.

    Logic: parse the AST, for each AsyncFunctionDef scan its source lines.
    If the function has >= 1 commented `await X` but 0 active `await X` → flag.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    lines = code.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue

        start = node.lineno - 1          # 0-indexed
        end = getattr(node, 'end_lineno', None)
        if end is None:
            continue

        func_lines = lines[start:end]
        has_active_await = False
        has_commented_await = False

        for line in func_lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                if re.search(r'\bawait\s+\w+', stripped):
                    has_commented_await = True
            else:
                if re.search(r'\bawait\s+\w+', stripped):
                    has_active_await = True

        if has_commented_await and not has_active_await:
            return True

    return False


def _has_hardcoded_return_stubs(code: str) -> bool:
    """Detect functions that do setup work but return a hardcoded constant string.

    Catches the pattern where real logic is replaced with a literal return value:

        def answer_index(interact, system, vopros, db, k):
            docs = db.similarity_search_with_score(vopros, k=k)
            context = " ".join([...])
            return "Это захардкоженный ответ"  # stub!

    Rules:
    - Function body (after stripping docstring) must have ≥ 2 statements.
    - The LAST statement must be ``return <non-empty string constant>``.
    - No other return in the function body (outside nested defs) returns a
      non-constant value — to avoid false-positives on functions that have
      a real early-exit before the stub.

    Single-statement functions are intentionally skipped (they may legitimately
    return a string, e.g. ``def get_prompt(): return "You are..."``).
    """
    try:
        tree = compile(
            code,
            '<precheck_hardcoded_return>',
            'exec',
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
    except SyntaxError:
        return False

    def _strip_docstring(body):
        if (body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], 'value', None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            return body[1:]
        return body

    def _collect_returns_shallow(stmts):
        """Collect Return nodes without recursing into nested function/class defs."""
        returns = []
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                returns.append(stmt)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pass  # skip nested definitions
            else:
                child_stmts = [n for n in ast.iter_child_nodes(stmt) if isinstance(n, ast.stmt)]
                returns.extend(_collect_returns_shallow(child_stmts))
        return returns

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        body = _strip_docstring(node.body)

        # Require at least 2 statements: some setup code + the stub return.
        if len(body) < 2:
            continue

        last = body[-1]
        if not isinstance(last, ast.Return):
            continue
        ret_val = last.value
        if ret_val is None:
            continue
        if not isinstance(ret_val, ast.Constant):
            continue
        # Only flag non-empty string constants (the classic hardcoded-answer pattern).
        if not isinstance(ret_val.value, str) or not ret_val.value.strip():
            continue

        # Safety check: if any earlier return in the function returns a real (non-constant)
        # value, this function likely uses conditional logic — don't flag it.
        earlier_returns = _collect_returns_shallow(body[:-1])
        has_real_earlier_return = any(
            ret.value is not None and not isinstance(ret.value, ast.Constant)
            for ret in earlier_returns
        )
        if has_real_earlier_return:
            continue

        return True

    return False


def _has_empty_kb_in_rag_call(code: str) -> bool:
    """Detect: RAG-prompt generation function called with an empty string as knowledge-base arg.

    Catches the pattern where a student passes '' (empty string) to generate_rag_prompt()
    instead of the actual knowledge_base variable, effectively disabling the knowledge base.
    e.g. full_prompt = generate_rag_prompt(user_question, '')
    """
    _RAG_PROMPT_FUNCS = frozenset({
        'generate_rag_prompt', 'generate_prompt', 'build_rag_prompt', 'build_prompt',
    })
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name not in _RAG_PROMPT_FUNCS:
            continue

        # Flag if any positional argument is an empty (or whitespace-only) string literal.
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and not arg.value.strip():
                return True

    return False


def run_precheck(raw_code: str) -> PrecheckResult:
    """
    Run deterministic precheck on raw code.
    
    Checks for:
    - Valid code presence
    - Syntax errors
    - Stubs (pass, TODO)
    - Metadata presence
    
    Args:
        raw_code: Raw code from notebook
        
    Returns:
        PrecheckResult with validation info
    """
    result = PrecheckResult(
        has_valid_code=True,
        has_syntax_errors=False,
        has_stubs=False,
        has_metadata=False,
        forced_score=None,
        reasons=[]
    )
    
    # Check if code is too short
    if not raw_code or len(raw_code.strip()) < 10:
        result.has_valid_code = False
        result.forced_score = 0
        result.reasons.append("Отсутствует или слишком короткий код решения.")
        return result
    
    # Check for metadata markers — only scan the code section, not the outputs
    # section (which may legitimately contain strings like "Outputs:").
    _OUTPUTS_MARKER = '# === CELL OUTPUTS ==='
    _code_section = raw_code[:raw_code.index(_OUTPUTS_MARKER)] if _OUTPUTS_MARKER in raw_code else raw_code
    metadata_markers = [
        "Cell type:",
        "Execution count:",
        "Outputs:",
        "Executed by:",
        "Executed at:"
    ]
    if any(marker in _code_section for marker in metadata_markers):
        result.has_metadata = True
        result.reasons.append("Код содержит служебные метаданные файла вместо чистого Python-кода.")
    
    # Clean and parse code for semantic checks.
    cleaned_code = clean_notebook_code(raw_code)

    # --- Syntax check on FULL cleaned code BEFORE parse_code truncates it ---
    # parse_code._split_code_and_trailing_output silently cuts the file at the first
    # syntax error, so validate_python_syntax(executable_code) would miss it.
    # We compile cleaned_code directly here and check only if the error line looks
    # like real Python code (to avoid false-positives from trailing notebook output).
    try:
        compile(cleaned_code, '<precheck_full>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except SyntaxError as e:
        lineno = e.lineno or 0
        code_lines = cleaned_code.split('\n')
        error_line = code_lines[lineno - 1] if 0 < lineno <= len(code_lines) else ''
        # Accept the error as real if:
        # (a) the error line is clearly a Python code construct, OR
        # (b) the error message describes an unclosed/unterminated token (EOF/EOL errors
        #     happen when the parser reaches the end of the file mid-literal/expression —
        #     these are always real syntax errors, not trailing-output false-positives)
        error_msg = (e.msg or '').lower()
        is_real_syntax_error = re.match(
            r'\s*(async\s+def|def\s|class\s|if\s|elif\s|for\s|while\s|try\s*:|with\s|@)',
            error_line,
        ) or re.search(
            r'eof|eol|scanning|unterminated|unexpected\s+eof|was\s+never\s+closed',
            error_msg,
        )
        if is_real_syntax_error:
            result.has_syntax_errors = True
            if result.forced_score is None or result.forced_score > 2:
                result.forced_score = 2
            result.reasons.append(
                f"Код содержит синтаксические ошибки: {e.msg} "
                f"(строка {lineno}): {(e.text or '').strip()}"
            )

    parsed = parse_code(cleaned_code)
    executable_code = parsed.executable_code
    
    if len(executable_code.strip()) < 10:
        result.has_valid_code = False
        result.forced_score = 0
        result.reasons.append("После очистки от метаданных и служебных строк код решения оказался почти пустым.")
        return result
    
    # Check for placeholder stubs.
    if _has_stub_patterns(executable_code):
        result.has_stubs = True
        if result.forced_score is None or result.forced_score > 2:
            result.forced_score = 2
        result.reasons.append("В коде обнаружены заглушки (pass или TODO) вместо реализации.")

    # Check for hardcoded constant returns in non-trivial functions.
    if _has_hardcoded_return_stubs(executable_code):
        result.has_stubs = True
        if result.forced_score is None or result.forced_score > 2:
            result.forced_score = 2
        result.reasons.append(
            "Обнаружена функция с нетривиальным телом, которая возвращает захардкоженную строку "
            "вместо результата реальных вычислений. Функция содержит код, но не использует его результат."
        )

    # Check for commented-out response calls (e.g. # await message.answer(...)).
    # If ALL send-response calls are commented out, the bot never replies to the user.
    if _has_commented_out_response_calls(cleaned_code):
        if result.forced_score is None or result.forced_score > 2:
            result.forced_score = 2
        result.reasons.append(
            "Все вызовы отправки ответа пользователю (message.answer/reply) закомментированы. "
            "Бот не отправляет ответ пользователю."
        )

    # Broader check: any async function where every await call is commented out.
    # Catches disabled async operations regardless of method name.
    if _async_func_has_only_commented_awaits(cleaned_code):
        if result.forced_score is None or result.forced_score > 2:
            result.forced_score = 2
        result.reasons.append(
            "Обнаружена async-функция, в которой все await-вызовы закомментированы. "
            "Функция не выполняет асинхронные операции."
        )

    # Check for empty string passed as knowledge-base to RAG-prompt function.
    # Catches logical error where knowledge_base is bypassed with '' placeholder.
    if _has_empty_kb_in_rag_call(cleaned_code):
        if result.forced_score is None or result.forced_score > 2:
            result.forced_score = 2
        result.reasons.append(
            "В вызов функции генерации RAG-промпта передана пустая строка вместо базы знаний. "
            "Требование «использовать базу знаний компании» не выполнено."
        )

    # Check syntax on executable code (without trailing notebook outputs).
    # This catches any remaining syntax issues not caught by the full-code check above.
    if not result.has_syntax_errors:
        syntax_check = validate_python_syntax(executable_code)
        if not syntax_check['valid']:
            result.has_syntax_errors = True
            if result.forced_score is None or result.forced_score > 2:
                result.forced_score = 2
            error_snippet = syntax_check['error'][:200] if syntax_check['error'] else ""
            result.reasons.append(f"Код содержит синтаксические ошибки: {error_snippet}")
    
    return result
