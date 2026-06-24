"""Code parsing service - separates executable code, logs, and metadata."""
import ast
import re
from typing import Dict, Any
from backend.models import ParsedCode


def parse_code(raw_code: str) -> ParsedCode:
    """
    Parse code into executable code, logs, and metadata.

    Two paths:
    1. Content has the explicit _CELL_OUTPUTS_MARKER → it was produced by
       cell-type-aware parsing (external service or .ipynb extractor).  Split
       on the marker and return the code section as-is — no preamble filtering,
       no heuristic cleaning.
    2. No marker → dirty fallback content.  Run heuristic split + preamble
       filtering via _extract_executable_code.
    """
    if _CELL_OUTPUTS_MARKER in raw_code:
        idx = raw_code.index(_CELL_OUTPUTS_MARKER)
        executable_code = raw_code[:idx].rstrip()
        logs_text = raw_code[idx + len(_CELL_OUTPUTS_MARKER):].strip()
        metadata = _extract_metadata(executable_code)
        return ParsedCode(
            executable_code=executable_code,
            logs=logs_text if logs_text else 'Нет информации о логах/выводе',
            metadata=metadata,
            raw_code=raw_code
        )

    # Dirty / fallback path: heuristic split + preamble filtering.
    code_part, trailing_logs = _split_code_and_trailing_output(raw_code)
    metadata = _extract_metadata(code_part)
    logs = _extract_logs(code_part, trailing_logs)
    executable_code = _extract_executable_code(code_part)
    return ParsedCode(
        executable_code=executable_code,
        logs=logs,
        metadata=metadata,
        raw_code=raw_code
    )


def _extract_metadata(code: str) -> Dict[str, Any]:
    """Extract metadata from code."""
    metadata = {}
    
    # Extract imports and determine version/requirements
    import_lines = []
    lines = code.split('\n')
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append(stripped)
    
    metadata['imports'] = import_lines
    
    # Try to find Python version comments
    version_match = re.search(r'# python[: ]*(\d+\.\d+)', code, re.IGNORECASE)
    if version_match:
        metadata['python_version'] = version_match.group(1)
    
    # Check for common libraries
    libraries = set()
    for imp in import_lines:
        if 'numpy' in imp:
            libraries.add('numpy')
        if 'pandas' in imp:
            libraries.add('pandas')
        if 'sklearn' in imp or 'scikit-learn' in imp:
            libraries.add('scikit-learn')
        if 'matplotlib' in imp:
            libraries.add('matplotlib')
        if 'tensorflow' in imp:
            libraries.add('tensorflow')
        if 'torch' in imp or 'pytorch' in imp:
            libraries.add('pytorch')
        if 'requests' in imp:
            libraries.add('requests')
    
    metadata['detected_libraries'] = list(libraries)
    
    return metadata


def _looks_like_output_line(line: str) -> bool:
    """Heuristic check for notebook execution output lines."""
    stripped = line.strip()
    if not stripped:
        return False

    output_patterns = [
        r'^(INFO|DEBUG|WARNING|WARN|ERROR|CRITICAL)\s*:',
        r'^Traceback \(most recent call last\):',
        r'^[A-Za-z_][A-Za-z0-9_]*Error\s*:',
        r'^(Collecting|Downloading|Installing|Requirement already satisfied):',
        r'^Successfully installed\s+',
        r'^\d+\s*/\s*\d+\s*\[',
        r'^https?://',
        r'^Received SIGINT signal$',
        r'^\d+\.\s+.+$',
    ]
    if any(re.match(pattern, stripped) for pattern in output_patterns):
        return True

    return False


def _can_parse_python_block(source: str) -> bool:
    """Return True when text can be parsed as Python (including notebook top-level await)."""
    # Strip IPython magic lines before compile() — they cause false SyntaxErrors.
    cleaned = '\n'.join(
        '' if l.lstrip().startswith('!') or l.lstrip().startswith('%') else l
        for l in source.splitlines()
    )
    try:
        compile(cleaned, '<parsed_code>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        return True
    except SyntaxError:
        return False


_CELL_OUTPUTS_MARKER = '# === CELL OUTPUTS ==='


def _split_code_and_trailing_output(raw_code: str) -> tuple[str, list[str]]:
    """Split clean code from leaked runtime output appended after code.

    Two strategies:
    1. If the explicit _CELL_OUTPUTS_MARKER is present (injected by the
       .ipynb extractor), split on it — accurate and cheap.
    2. Otherwise scan backwards from the END to trim consecutive trailing
       lines that look like execution output.  We intentionally do NOT
       scan forward looking for the last parseable Python position, because
       that strategy incorrectly cuts the split when the student's code
       contains any non-Python text (e.g. a bare label without a '#') or
       genuine syntax errors before the actual execution output.
    """
    # Fast path: explicit marker inserted during .ipynb extraction.
    if _CELL_OUTPUTS_MARKER in raw_code:
        idx = raw_code.index(_CELL_OUTPUTS_MARKER)
        code_part = raw_code[:idx].rstrip()
        outputs_text = raw_code[idx + len(_CELL_OUTPUTS_MARKER):]
        trailing = [l for l in outputs_text.split('\n') if l.strip()]
        return code_part, trailing

    lines = raw_code.split('\n')
    if not lines:
        return raw_code, []

    # Backward scan: find the start of consecutive trailing output lines.
    # Stop as soon as we hit a line that does NOT look like execution output.
    trailing_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue  # blank lines — keep scanning
        if _looks_like_output_line(lines[i]):
            trailing_start = i
        else:
            break  # first non-output line from the end — this is code

    if trailing_start < len(lines):
        trailing = [l for l in lines[trailing_start:] if l.strip()]
        return '\n'.join(lines[:trailing_start]), trailing

    return raw_code, []


def _extract_logs(code: str, trailing_logs: list[str] | None = None) -> str:
    """
    Extract output/log information from code comments.
    
    Looks for:
    - Commented out output
    - Print statement documentation
    - Expected results
    """
    log_lines = []
    lines = code.split('\n')
    
    for line in lines:
        stripped = line.strip()
        
        # Look for "# Output:", "# Expected:", "# Result:" comments
        if any(marker in stripped for marker in ['# Output:', '# Expected:', '# Result:', '# Outputs:', '# Log:']):
            log_lines.append(line)
        
        # Look for print statements with strings (potential output documentation)
        if 'print(' in stripped and ('Output' in line or 'Expected' in line or 'Result' in line):
            log_lines.append(line)

        # Collect explicit raw execution outputs that leaked from notebook cells
        if _looks_like_output_line(line):
            log_lines.append(line)

    if trailing_logs:
        log_lines.extend(trailing_logs)

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for line in log_lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    
    return '\n'.join(deduped) if deduped else "Нет информации о логах/выводе"


def _extract_executable_code(code: str) -> str:
    """
    Extract clean executable code.
    
    Removes:
    - Output comments
    - Long comment blocks that look like documentation
    - Metadata comments
    
    Args:
        code: Raw code
        
    Returns:
        Clean executable code
    """
    lines = code.split('\n')
    executable_lines = []

    # Detect the preamble region: lines that appear BEFORE the first real
    # Python construct (import / from / def / class / async / decorator).
    # In this region we strip bare non-Python text labels (e.g. "код написан
    # ChatGPT") that come from markdown cells or student cell annotations and
    # would otherwise cause false-positive SyntaxError flags in the precheck.
    _PY_STARTERS = ('import ', 'from ', 'def ', 'class ', 'async ', '@')
    preamble_end = len(lines)
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if s and any(s.startswith(p) for p in _PY_STARTERS):
            preamble_end = idx
            break

    skip_next_lines = 0
    for i, line in enumerate(lines):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue

        # Stop at the explicit cell-outputs separator — everything after it
        # is execution output, not executable code.
        if line.strip() == _CELL_OUTPUTS_MARKER:
            break
        
        stripped = line.strip()

        # In the preamble region (before the first Python construct) strip bare
        # non-Python text lines that are neither a comment nor valid Python syntax.
        # Typical example: "код написан ChatGPT" from a student label / markdown
        # cell that ends up in the flat text returned by the external service.
        # IMPORTANT: only filter lines that have no Python operator characters —
        # multi-line Python expressions (e.g. questions = ['...', on line 1) are
        # syntactically incomplete when compiled in isolation and would be wrongly
        # dropped if we tried compile() on every preamble line.
        if i < preamble_end and stripped and not stripped.startswith('#'):
            _PY_OPERATORS = set('=([{:\'\"+-*/%|&!<>@')
            if not any(c in stripped for c in _PY_OPERATORS):
                try:
                    compile(stripped, '<preamble_check>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                except SyntaxError:
                    continue  # non-Python label — drop it

        # Skip comment-only lines that look like output documentation
        if stripped.startswith('#'):
            # Skip lines like "# Output:", "# Expected:", etc.
            if any(marker in stripped for marker in ['# Output:', '# Expected:', '# Result:', '# Outputs:']):
                # Skip this and next few lines if they're also comments
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('#'):
                    skip_next_lines += 1
                    j += 1
                continue

        # Skip leaked execution outputs/log lines
        if _looks_like_output_line(line):
            continue
        
        executable_lines.append(line)
    
    result = '\n'.join(executable_lines)

    # Keep code unchanged if not parseable: generator/evaluator can still inspect it.
    
    # Clean up excessive empty lines
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    
    return result.strip()
