"""Shared deterministic checks for RAG pipeline correctness.

Used by both the evaluator (to generate semantic hints) and the generator
validator (to gate incorrect cases where the described bug was not applied).
"""
import ast
import re as _re

# Vector-search method names whose results must reach the LLM messages list.
RAG_SEARCH_METHODS = frozenset({
    'similarity_search',
    'similarity_search_with_score',
    'max_marginal_relevance_search',
    'get_relevant_documents',
})

# Vector-store construction / retriever methods. Their presence is a reliable
# signal that the code uses a *vector-store* RAG mechanism that our static
# checks can actually analyse — as opposed to web-search agents (Tavily,
# SerpAPI) or tool-based agents whose `.run()/.invoke()` calls we cannot
# reliably trace. Used to gate the generator pre-gate so it never second-guesses
# the LLM validator on code our checks don't understand.
_VECTOR_STORE_BUILDERS = frozenset({
    'from_documents',
    'from_texts',
    'from_embeddings',
    'as_retriever',
})


# Keywords in a change_summary that signal a RAG pipeline bug claim.
# IMPORTANT: keep these narrow — overly broad patterns cause false positives
# on non-RAG code (e.g. "инструмент не включён в tools" matches r'не включен').
_RAG_SUMMARY_PATTERNS = (
    r'similarity_search',           # direct LangChain/FAISS API name
    r'similarity\s+search',         # English two-word form
    r'get_relevant_documents',      # LangChain retriever method
    r'max_marginal_relevance',      # another FAISS method
    r'после создания сообщений',    # specific RAG ordering issue phrase
    r'результаты поиска не',        # "results of search not..."
    r'поиска не использ',           # "search not used..."
    r'без.*базы знаний',            # "without knowledge base"
    r'базы знаний.*не',             # "knowledge base not..."
    r'пустой список.*вместо',       # "empty list instead of (docs)"
    r'возвращает пустой',           # "returns empty (list/result)"
    r'empty list',                  # English form
    r'векторн.*поиск',              # "vector search"
    r'vectorstore.*не',             # "vectorstore not..."
)


def check_rag_results_not_in_messages(tree: ast.AST) -> list[str]:
    """Return КРИТИЧНО hints for every function where RAG search results do not
    flow into the ``messages`` list passed to the LLM.

    Returns an empty list when no issue is detected (code is correct).

    Detection algorithm
    -------------------
    1. Collect variables directly assigned from RAG search calls (e.g. ``docs``).
    2. Compute the transitive closure of derived variables (e.g. ``docs`` →
       ``message_content`` → ``system_prompt``).
    3. Find the ``messages = [...]`` assignment(s) in the function body.
    4. Flag the function if none of the derived variables appear inside the
       messages list, and no ``messages.append(...)`` with a derived variable
       follows later.
    """
    issues: list[str] = []

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # ── Step 1: variables directly from RAG search calls ─────────────────
        search_result_vars: set[str] = set()
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                for sub in ast.walk(node.value):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr in RAG_SEARCH_METHODS):
                        search_result_vars.add(target.id)

        if not search_result_vars:
            continue

        # ── Step 2: transitive closure of derived variables ───────────────────
        derived_vars: set[str] = set()
        prev_size = -1
        all_rag_vars = search_result_vars | derived_vars
        while len(derived_vars) != prev_size:
            prev_size = len(derived_vars)
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id in all_rag_vars:
                        continue
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.Name) and sub.id in all_rag_vars:
                            derived_vars.add(target.id)
                            break
            all_rag_vars = search_result_vars | derived_vars

        # ── Step 3: find `messages = [...]` assignments ──────────────────────
        messages_stmts: list[ast.Assign] = [
            stmt for stmt in func_node.body
            if isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'messages' for t in stmt.targets)
        ]
        if not messages_stmts:
            continue

        # ── Step 4: check each messages assignment ────────────────────────────
        for msg_stmt in messages_stmts:
            rag_in_messages = any(
                isinstance(n, ast.Name) and n.id in all_rag_vars
                for n in ast.walk(msg_stmt.value)
            )
            if rag_in_messages:
                continue

            appended = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('append', 'insert', 'extend')
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'messages'
                and stmt.lineno > msg_stmt.lineno
                and any(isinstance(s, ast.Name) and s.id in all_rag_vars
                        for s in ast.walk(node))
                for stmt in func_node.body
                for node in ast.walk(stmt)
            )
            if appended:
                continue

            derived_str = ', '.join(sorted(derived_vars)) if derived_vars else 'нет'
            issues.append(
                f"КРИТИЧНО: в функции `{func_node.name}` результаты поиска по векторной базе "
                f"({', '.join(sorted(search_result_vars))}) вычислены и преобразованы "
                f"(производные переменные: {derived_str}), "
                "но НИ ОДНА из них не включена в список `messages`, который передаётся в LLM. "
                "Ответ LLM генерируется без контекста из базы знаний — "
                "RAG-цепочка разорвана, требование использовать результаты поиска фактически не выполнено."
            )

    return issues


def check_empty_source_in_messages(tree: ast.AST) -> list[str]:
    """Detect when messages content is built from an empty collection — logic-based, no name heuristics.

    Algorithm (pure data-flow, variable names don't matter):

    1. Find variables assigned to an empty list ``[]`` within the function.
    2. Of those, identify which are never populated by a RAG search call
       AND never appended to (which would be the conditional-fill pattern).
    3. Compute the transitive closure of variables derived from those empty vars.
    4. Flag the function if any of those derived variables appear inside the
       ``messages`` list passed to the LLM.

    This catches patterns like:
        docs = []                                         # empty — no search
        ctx  = "\\n".join(d.page_content for d in docs)  # always ""
        messages = [{"role": "user", "content": f"...{ctx}..."}]  # empty context
    regardless of what variables are called.
    """
    issues: list[str] = []

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # ── Step 1: variables assigned to empty list [] ───────────────────────
        empty_vars: set[str] = set()
        for node in ast.walk(func_node):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.List)
                    and len(node.value.elts) == 0):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        empty_vars.add(target.id)

        if not empty_vars:
            continue

        # ── Step 2: keep only those never re-populated by search or append ────
        unpopulated: set[str] = set()
        for var in empty_vars:
            # Reassigned from a RAG search call?
            search_reassigns = any(
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == var for t in stmt.targets)
                and any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in RAG_SEARCH_METHODS
                    for sub in ast.walk(stmt.value)
                )
                for stmt in ast.walk(func_node)
                if isinstance(stmt, ast.Assign)
            )
            # Ever appended / extended?
            appended = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('append', 'extend', 'insert')
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var
                for node in ast.walk(func_node)
            )
            if not search_reassigns and not appended:
                unpopulated.add(var)

        if not unpopulated:
            continue

        # ── Step 3: transitive closure of variables derived from empty vars ───
        derived: set[str] = set(unpopulated)
        prev_size = -1
        while len(derived) != prev_size:
            prev_size = len(derived)
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name) or target.id in derived:
                        continue
                    if any(isinstance(s, ast.Name) and s.id in derived
                           for s in ast.walk(node.value)):
                        derived.add(target.id)

        # ── Step 4: check if any derived var appears in messages ─────────────
        messages_stmts = [
            stmt for stmt in func_node.body
            if isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'messages' for t in stmt.targets)
        ]
        for msg_stmt in messages_stmts:
            empty_in_messages = any(
                isinstance(n, ast.Name) and n.id in derived
                for n in ast.walk(msg_stmt.value)
            )
            if empty_in_messages:
                issues.append(
                    f"КРИТИЧНО: в функции `{func_node.name}` содержимое `messages` "
                    f"формируется из переменных ({', '.join(sorted(unpopulated))}), "
                    "которые инициализированы пустыми коллекциями (`[]`) "
                    "и ни разу не заполнялись результатами поиска по базе знаний. "
                    "LLM получает пустой контекст вместо релевантных документов — "
                    "RAG-пайплайн сломан."
                )
                break

    return issues


def code_uses_vector_store_rag(tree: ast.AST) -> bool:
    """Return True if the code uses a vector-store RAG mechanism our checks understand.

    Detects either:
      - a direct vector-search call (similarity_search, get_relevant_documents, …), or
      - vector-store construction / retriever setup
        (FAISS.from_documents, db.as_retriever, …).

    Returns False for web-search / tool agents (Tavily, SerpAPI) and any code
    whose search mechanism our static analysis cannot trace. The generator
    pre-gate relies on this: it must NOT reject a logical_error case based on
    our checks finding "no issue" when those checks are simply not applicable.
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (node.func.attr in RAG_SEARCH_METHODS
                     or node.func.attr in _VECTOR_STORE_BUILDERS)):
            return True
    return False


def check_all_rag_issues(tree: ast.AST) -> list[str]:
    """Run all RAG pipeline checks and return combined list of issues."""
    return check_rag_results_not_in_messages(tree) + check_empty_source_in_messages(tree)


def summary_claims_rag_issue(change_summary: str) -> bool:
    """Return True when a change_summary text claims any kind of RAG pipeline bug."""
    text = (change_summary or '').lower()
    return any(_re.search(pattern, text) for pattern in _RAG_SUMMARY_PATTERNS)
