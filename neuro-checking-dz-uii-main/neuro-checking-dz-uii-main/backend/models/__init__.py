"""Data models and schemas for the application."""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class ErrorType(str, Enum):
    """Types of errors that can be generated."""
    SYNTAX_ERROR = "syntax_error"
    LOGICAL_ERROR = "logical_error"
    NON_OPTIMAL = "non_optimal"
    PARTIAL = "partial"
    CHEATING = "cheating"


@dataclass
class PrecheckResult:
    """Result of deterministic precheck."""
    has_valid_code: bool
    has_syntax_errors: bool
    has_stubs: bool
    has_metadata: bool
    forced_score: Optional[int] = None
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class ParsedCode:
    """Parsed code structure."""
    executable_code: str  # Clean executable code
    logs: str             # Output/logs from execution
    metadata: Dict[str, Any]  # Metadata, version info, etc
    raw_code: str         # Original raw code


@dataclass
class GeneratedCase:
    """A generated test case."""
    type: str  # "perfect_etalon", "perfect_alternative", "syntax_error", etc
    student_code: str
    expected_grade: int
    change_summary: str
    execution_logs: Optional[str] = None  # synthetic logs/conclusions for this case


@dataclass
class EvaluationResult:
    """Result of LLM evaluation."""
    topic: str
    overall_score: int
    overall_comment: str
    homework_tasks: List[Dict[str, Any]]
    additional_recommendations: str
    pre_check: Dict[str, Any]
    usage: Dict[str, int]  # token usage


@dataclass
class ColabLoadRequest:
    """Request to load and parse Colab content."""
    url: str


@dataclass
class CheckHomeworkRequest:
    """Request to check homework."""
    tema: str
    zadanie: str
    resheniye: str
    name: str = "Участник"
    model: str = "gpt-4o"


@dataclass
class GenerateCasesRequest:
    """Request to generate test cases."""
    etalon_link: str
    tema: str
    zadanie: str
    num_correct: int
    num_incorrect: int
    model: str = "gpt-4o"
    additional_colabs: Optional[List[Dict[str, Any]]] = None
    enabled_types: Optional[List[str]] = None
