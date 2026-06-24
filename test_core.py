#!/usr/bin/env python3
"""Quick test of core services."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.code_cleaner import clean_notebook_code
from backend.services.precheck import run_precheck
from backend.services.code_parser import parse_code
from backend.utils.syntax_validator import validate_python_syntax
from backend.utils.patch_applier import apply_patches


def test_code_cleaner():
    """Test code cleaning."""
    print("\n=== Testing Code Cleaner ===")
    
    raw_code = """
    Cell type: code
    Execution count: 1
    
    import numpy as np
    print("Hello")
    
    # Output: Hello
    """
    
    cleaned = clean_notebook_code(raw_code)
    print(f"Input length: {len(raw_code)}")
    print(f"Output length: {len(cleaned)}")
    print(f"Output:\n{cleaned}\n")
    assert len(cleaned) > 0
    assert "Cell type" not in cleaned
    print("✓ Code cleaner works!")


def test_precheck():
    """Test precheck validation."""
    print("\n=== Testing Precheck ===")
    
    # Valid code
    code_valid = "x = 1\nprint(x)"
    result = run_precheck(code_valid)
    print(f"Valid code: has_valid_code={result.has_valid_code}, forced_score={result.forced_score}")
    assert result.has_valid_code
    assert result.forced_score is None
    print("✓ Valid code passed!")
    
    # Code with stub
    code_stub = "def foo():\n    pass"
    result = run_precheck(code_stub)
    print(f"Stub code: has_stubs={result.has_stubs}, forced_score={result.forced_score}")
    assert result.has_stubs
    assert result.forced_score == 2
    print("✓ Stub detection works!")
    
    # Too short
    code_short = "x"
    result = run_precheck(code_short)
    print(f"Short code: has_valid_code={result.has_valid_code}, forced_score={result.forced_score}")
    assert not result.has_valid_code
    assert result.forced_score == 0
    print("✓ Short code detection works!")


def test_code_parser():
    """Test code parsing."""
    print("\n=== Testing Code Parser ===")
    
    code = """
import pandas as pd
import numpy as np

# Create data
df = pd.DataFrame({'a': [1, 2, 3]})
print(df)

# Output: DataFrame with column 'a'
"""
    
    parsed = parse_code(code)
    print(f"Executable code lines: {len(parsed.executable_code.split(chr(10)))}")
    print(f"Metadata imports: {len(parsed.metadata['imports'])}")
    print(f"Detected libraries: {parsed.metadata['detected_libraries']}")
    
    assert len(parsed.executable_code) > 0
    assert 'pandas' in parsed.metadata['detected_libraries']
    assert 'numpy' in parsed.metadata['detected_libraries']
    print("✓ Code parser works!")


def test_syntax_validator():
    """Test syntax validation."""
    print("\n=== Testing Syntax Validator ===")
    
    # Valid syntax
    valid_code = "x = 1\nprint(x)"
    result = validate_python_syntax(valid_code)
    print(f"Valid code: {result}")
    assert result['valid']
    print("✓ Valid syntax recognized!")
    
    # Invalid syntax
    invalid_code = "if True\n    pass"  # Missing colon
    result = validate_python_syntax(invalid_code)
    print(f"Invalid code: {result['valid']}, error: {result['error'][:50] if result['error'] else 'None'}")
    assert not result['valid']
    print("✓ Invalid syntax detected!")


def test_patch_applier():
    """Test patch application."""
    print("\n=== Testing Patch Applier ===")
    
    code = """
def hello():
    print("world")
"""
    
    patches = [
        {
            "find": 'print("world")',
            "replace": 'print("hello")'
        }
    ]
    
    result = apply_patches(code, patches)
    print(f"Patch success: {result['success']}")
    print(f"Result contains 'hello': {'hello' in result['result']}")
    
    assert result['success']
    assert 'hello' in result['result']
    print("✓ Patch applier works!")


if __name__ == '__main__':
    print("🧪 Running core service tests...\n")
    
    try:
        test_code_cleaner()
        test_precheck()
        test_code_parser()
        test_syntax_validator()
        test_patch_applier()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
