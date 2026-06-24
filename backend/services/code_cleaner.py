"""Code cleaning service - removes metadata, ANSI codes, and service lines."""
import re


def clean_notebook_code(code: str) -> str:
    """
    Removes notebook metadata, ANSI escapes, and service lines.
    
    Args:
        code: Raw code from notebook
        
    Returns:
        Clean executable Python code
        
    Raises:
        ValueError: If code appears to be HTML or otherwise invalid
    """
    if not code:
        return ""

    # Some corrupted downloads may contain NUL bytes; strip them early.
    code = code.replace('\x00', '')
    
    # Early check: reject HTML/XML content
    stripped = code.strip()
    if stripped.startswith('<!DOCTYPE') or stripped.startswith('<html') or stripped.startswith('<?xml'):
        raise ValueError("Content appears to be HTML/XML, not Python code")
    if '<head>' in stripped or '<script' in stripped or '<style' in stripped or '<body' in stripped:
        raise ValueError("Content contains HTML tags, not Python code")
    
    metadata_markers = [
        "Cell type:",
        "Execution count:",
        "Executed:",
        "Executed by:",
        "Executed at:",
        "Outputs:",
        "# --- Cell ---",
        "# Содержимое файла",
        "---"
    ]
    
    # Check if line is a separator (5+ equal signs)
    def is_separator(s: str) -> bool:
        return bool(re.match(r'^[ \t]*={5,}[ \t]*$', s))
    
    lines = code.split('\n')
    
    # 1. Filter lines and clean prefixes
    processed_lines = []
    in_shell_continuation = False  # True while inside a multi-line !cmd \ or %cmd \ block
    for line in lines:
        current_line = line
        
        # Remove "Source: " prefix if it exists
        trimmed = current_line.strip()
        if trimmed.startswith("Source:"):
            current_line = re.sub(r'^[ \t]*Source:[ \t]*', '', current_line)
        
        trimmed = current_line.strip()
        
        # If effectively empty, preserve as empty line
        if not trimmed:
            if in_shell_continuation:
                # blank line always ends a shell continuation block
                in_shell_continuation = False
            processed_lines.append("")
            continue
        
        # Remove specific metadata markers
        if any(marker in trimmed for marker in metadata_markers):
            if any(trimmed.startswith(marker) for marker in metadata_markers):
                continue
        
        # Remove separator lines
        if is_separator(trimmed):
            continue
        
        # Remove shell commands and magic commands (including multi-line continuations)
        if trimmed.startswith('!') or trimmed.startswith('%'):
            # If line ends with \, the next lines are continuations of this shell command
            in_shell_continuation = trimmed.endswith('\\')
            continue
        
        # Skip continuation lines of a multi-line shell command (e.g. indented package names)
        if in_shell_continuation:
            in_shell_continuation = trimmed.endswith('\\')
            continue
        
        # Remove isolated ANSI sequences like [?25h, [?25l
        if re.match(r'^\[(?:\?|[0-9;]+)[a-zA-Z]$', trimmed):
            continue
        
        processed_lines.append(current_line)
    
    # 2. Join and remove ANSI escape sequences
    cleaned = '\n'.join(processed_lines)
    
    # Remove ANSI escape sequences
    cleaned = re.sub(
        r'[\u001b\u009b][\[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]',
        '',
        cleaned
    )
    
    # 3. Remove isolated bracket sequences that might have lost their ESC char
    cleaned = re.sub(r'\[(?:\?|[0-9;]+)[0-9;]*[a-zA-Z]', '', cleaned)
    
    # 4. Remove excessive internal empty lines (3+ to 2)
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
    
    # 5. Final trim
    return cleaned.strip()
