"""
Data models for gaps and questions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# Scenario kinds (fine-grained labels for template selection at scale)
SCENARIO_KIND_ALREADY_CLOSED_RETURN = "already_closed_return"
SCENARIO_KIND_NOT_INITIALIZED_CALLS_INIT = "not_initialized_calls_init"
SCENARIO_KIND_NULL_KEYARGS_BRANCH = "null_keyArgs_branch"
SCENARIO_KIND_UNGUARDED_PARSE_FAILURE = "unguarded_parse_failure"
SCENARIO_KIND_BROAD_THROWS_MISMATCH = "broad_throws_mismatch"
SCENARIO_KIND_RESOURCE_CLEANUP_FINALLY = "resource_cleanup_finally"
# Coarse kinds (fallback when no fine-grained match)
SCENARIO_KIND_EARLY_RETURN = "early_return"
SCENARIO_KIND_CONDITIONAL_BRANCH = "conditional_branch"
SCENARIO_KIND_EXCEPTION_PATH = "exception_path"
SCENARIO_KIND_STATE_DEPENDENT = "state_dependent"

# Outcome tokens for template/validator (options may only mention these when present)
SCENARIO_OUTCOME_RETURN = "return"
SCENARIO_OUTCOME_THROW = "throw"
SCENARIO_OUTCOME_SIDE_EFFECTS = "side effects"
SCENARIO_OUTCOME_LOGGING = "logging"
SCENARIO_OUTCOME_CLEANUP = "cleanup"
SCENARIO_OUTCOME_COMMIT = "commit"
SCENARIO_OUTCOME_STATE_MUTATION = "state mutation"


@dataclass
class ScenarioSpec:
    """
    Structured scenario candidate for data-driven template selection.
    Built per method from AST/detector; used to select template and generate options.
    """
    scenario_id: str
    scenario_trigger_type: str  # early_return, conditional_branch, exception_path, state_dependent
    scenario_kind: str  # fine-grained: already_closed_return, not_initialized_calls_init, null_keyArgs_branch, ...
    scenario_condition: str
    scenario_evidence: str  # snippet + optional AST node ids
    scenario_outcomes: List[str]  # return, throw, side effects, logging, cleanup, commit, state mutation
    evidence_confidence: str  # high, medium, low
    code_determined: bool  # True if outcome is fully implied by code (-> auto_add, do not ask)


@dataclass
class Gap:
    """Represents a detected gap in documentation."""
    id: str
    type: str  # e.g., "missing_precondition", "missing_exception", "field_write_fact", etc.
    doc_slot: str  # e.g., "Preconditions", "Exceptions", "SideEffects"
    priority: int  # 1-5, higher = more critical
    evidence_confidence: str  # "high", "medium", "low"
    kind: str  # "fact" or "guarantee" (renamed from fact_or_guarantee)
    action: str = ""  # "auto_add", "ask", "skip" - how to handle this gap
    parameters: List[str] = field(default_factory=list)
    issue: str = ""
    evidence_snippet: str = ""
    question: str = ""
    suggested_options: List[Dict[str, str]] = field(default_factory=list)
    dedup_key: str = ""
    
    # Enhanced fields for research alignment
    doc_insert_target: str = ""  # Where to insert answer: "Preconditions", "Returns", "SideEffects", etc.
    risk_level: str = "medium"  # "high", "medium", "low" - bug likelihood
    llm_rationale: Optional[str] = None  # LLM's explanation for why this question is important
    llm_rank: Optional[int] = None  # LLM's ranking (1 = highest priority) for analysis
    
    # Execution scenario gap fields (machine-readable; do not lose during LLM rewriting)
    scenario_kind: Optional[str] = None  # Fine-grained: already_closed_return, not_initialized_calls_init, null_keyArgs_branch, ...
    scenario_condition: Optional[str] = None  # Precise condition, e.g. "closed == true at entry", "keyArgs == null"
    scenario_trigger_type: Optional[str] = None  # early_return, conditional_branch, exception_path, state_dependent
    scenario_expected_effects: Optional[str] = None  # Comma-separated outcomes (legacy)
    scenario_outcomes: Optional[List[str]] = None  # [return, throw, logging, cleanup, commit, ...]; options may only mention these
    scenario_evidence: Optional[str] = None  # Snippet + optional AST node ids
    code_determined: Optional[bool] = None  # True -> auto_add, do not ask


@dataclass
class Question:
    """Represents a question for developers."""
    id: str
    priority: int
    category: str
    doc_slot: str
    question_text: str
    context_code: str
    options: List[Dict[str, str]] = field(default_factory=list)
    evidence_confidence: str = "high"
    fact_or_guarantee: str = "guarantee"
    developer_answer: Optional[str] = None
    answered: bool = False
    # Scenario question logging (for reproducibility at scale)
    scenario_kind: Optional[str] = None
    scenario_condition: Optional[str] = None
    scenario_template_id: Optional[str] = None
    template_registry_version: Optional[str] = None

