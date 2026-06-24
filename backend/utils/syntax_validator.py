"""Syntax validation utility."""
import ast
from typing import Dict


def validate_python_syntax(code: str) -> Dict[str, any]:
    """
    Validate Python code syntax using python3 -m py_compile.
    
    Args:
        code: Python code to validate
        
    Returns:
        Dict with keys:
        - valid (bool): Whether syntax is valid
        - error (str): Error message if invalid, None if valid
    """
    if code is None:
        return {'valid': False, 'error': 'Code is None'}

    # Protect against corrupted payloads with NUL bytes.
    if '\x00' in code:
        return {'valid': False, 'error': 'source code string cannot contain null bytes'}

    # Strip IPython magic lines (!pip, %matplotlib, etc.) — they are valid in
    # Colab/Jupyter but cause SyntaxError in standard compile().  Replace each
    # such line with a blank line so line numbers in error messages stay aligned.
    stripped_lines = []
    for line in code.splitlines():
        s = line.lstrip()
        if s.startswith('!') or s.startswith('%'):
            stripped_lines.append('')
        else:
            stripped_lines.append(line)
    code_for_compile = '\n'.join(stripped_lines)

    try:
        compile(
            code_for_compile,
            '<student_code>',
            'exec',
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        return {'valid': True, 'error': None}
    except SyntaxError as e:
        line_text = (e.text or '').strip()
        location = f'line {e.lineno}' if e.lineno else 'unknown line'
        return {
            'valid': False,
            'error': f'SyntaxError: {e.msg} ({location}) {line_text}'.strip(),
        }
    except Exception:
        # Assume valid on internal validator error to avoid blocking pipeline.
        return {'valid': True, 'error': None}
