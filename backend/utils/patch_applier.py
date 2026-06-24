"""Patch application utility."""
import re
from typing import Dict, List


def apply_patches(code: str, patches: List[Dict[str, str]]) -> Dict:
    """
    Apply patches (find/replace pairs) to code.
    
    Handles various whitespace formats to find matches robustly.
    
    Args:
        code: Original code
        patches: List of dicts with 'find' and 'replace' keys
        
    Returns:
        Dict with keys:
        - success (bool): Whether all patches applied
        - result (str): Code after patches (or original if failed)
        - error (str): Error message if failed, None if successful
    """
    if not patches:
        return {'success': False, 'result': code, 'error': 'No patches provided'}
    
    current_code = code
    
    def normalize_ws(s: str) -> str:
        """Normalize whitespace for comparison."""
        return '\n'.join(line.rstrip() for line in s.split('\n')).strip()
    
    for patch in patches:
        if not patch.get('find'):
            continue
        
        find_str = patch['find']
        replace_str = patch['replace']
        
        # Attempt 1: Exact match
        occurrences = current_code.count(find_str)
        if occurrences == 1:
            current_code = current_code.replace(find_str, replace_str)
            continue
        elif occurrences > 1:
            # Multiple exact matches — apply to the first occurrence only.
            # This handles notebooks where the same function is defined in multiple cells.
            current_code = current_code.replace(find_str, replace_str, 1)
            continue
        
        # Attempt 2: Try with normalized whitespace
        normalized_code = normalize_ws(current_code)
        normalized_find = normalize_ws(find_str)
        
        norm_occurrences = normalized_code.count(normalized_find)
        
        if norm_occurrences >= 1:
            # Apply using regex with flexible whitespace — count=1 ensures first match only
            escaped_find = re.escape(normalized_find)
            # Replace newlines with flexible whitespace pattern
            pattern = escaped_find.replace(r'\n', r'\s*\n\s*')
            regex = re.compile(pattern, re.MULTILINE)
            
            if regex.search(current_code):
                current_code = regex.sub(replace_str, current_code, count=1)
                continue
            return {
                'success': False,
                'result': code,
                'error': f'Patch target not found: "{find_str[:50]}..."'
            }
    
    return {'success': True, 'result': current_code, 'error': None}
