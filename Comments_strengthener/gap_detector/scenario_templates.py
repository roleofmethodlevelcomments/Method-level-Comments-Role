"""
Data-driven scenario template registry for scale (hundreds of methods).

Selection order: scenario_kind in TEMPLATE_REGISTRY -> use that template;
else fallback by scenario_trigger_type. Templates take (spec, method_facts) and return options.
Options mention only effects present in code (method_facts) and scenario_outcomes;
contradicting options get "intended differs"; always include an escape (Not specified / Other).
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
from gap_detector.models import (
    ScenarioSpec,
    SCENARIO_KIND_ALREADY_CLOSED_RETURN,
    SCENARIO_KIND_NOT_INITIALIZED_CALLS_INIT,
    SCENARIO_KIND_NULL_KEYARGS_BRANCH,
    SCENARIO_KIND_UNGUARDED_PARSE_FAILURE,
    SCENARIO_KIND_EARLY_RETURN,
    SCENARIO_KIND_CONDITIONAL_BRANCH,
    SCENARIO_KIND_EXCEPTION_PATH,
    SCENARIO_KIND_STATE_DEPENDENT,
)

# For reproducibility at dataset scale
TEMPLATE_REGISTRY_VERSION = "1.1"

# Type: (spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]
TemplateFn = Callable[[ScenarioSpec, Dict[str, Any]], List[Dict[str, str]]]


def _doc_slot(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> str:
    return method_facts.get("doc_slot", "Postconditions")


def _condition_entity(condition: str) -> str:
    """Extract a short entity from condition for option wording (e.g. 'userUrl' from 'userUrl == null')."""
    if not condition or len(condition) > 60:
        return ""
    c = condition.strip()
    for sep in (" == null", "!= null", " is null", " is not null", " == null)", "!= null)"):
        if sep in c.lower():
            return c.split(sep)[0].strip().split()[-1] or ""
    return ""


def template_already_closed_return(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    has_cleanup = method_facts.get("has_cleanup", False)
    if has_cleanup:
        a_text = "Returns immediately; cleanup/close is not run."
        c_text = "Performs close/cleanup anyway, intended differs"
    else:
        a_text = "Returns immediately without side effects"
        c_text = "Performs main path anyway, intended differs"
    return [
        {"key": "A", "text": a_text, "doc_insert_target": doc_slot},
        {"key": "B", "text": "Throws exception instead, intended differs", "doc_insert_target": doc_slot},
        {"key": "C", "text": c_text, "doc_insert_target": doc_slot},
        {"key": "D", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


def template_not_initialized_calls_init(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    has_init = method_facts.get("has_init_call", True)
    a_text = (
        "init is called when not initialized. If init fails, exception propagates. Otherwise returns value."
        if has_init
        else "When not initialized, method throws or returns; document which."
    )
    return [
        {"key": "A", "text": a_text, "doc_insert_target": doc_slot},
        {"key": "B", "text": "Returns fallback value on init failure, intended differs from current code.", "doc_insert_target": doc_slot},
        {"key": "C", "text": "Not specified. Document that init is invoked and exceptions propagate.", "doc_insert_target": doc_slot},
    ]


def template_null_keyArgs_branch(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    outcomes = set(o.lower() for o in (spec.scenario_outcomes or []))
    has_commit = method_facts.get("has_commit", False)
    has_cleanup = method_facts.get("has_cleanup", False)
    has_warn = method_facts.get("has_warn", False)
    branch = method_facts.get("branch_outcome") or {}

    # Dynamic A/B/C from branch outcome when known
    abc = _options_abc_from_branch_outcome(branch)
    if abc is not None:
        a_text, b_text, c_text = abc
    else:
        parts = []
        if has_warn and "logging" in outcomes:
            parts.append("Logs and skips")
        if has_commit and "commit" in outcomes:
            parts.append("skips commit")
        if has_cleanup and "cleanup" in outcomes:
            parts.append("cleanup runs")
        if parts:
            a_text = "; ".join(parts).strip()
            if not a_text.endswith("."):
                a_text += "."
        else:
            entity = _condition_entity(spec.scenario_condition or "")
            a_text = f"Takes branch when {entity} is null; document behavior." if entity else "Takes branch; document behavior."
        b_text = "Throws exception on null, intended differs"
        c_text = "Attempts main path with null, intended differs"

    return [
        {"key": "A", "text": a_text, "doc_insert_target": doc_slot},
        {"key": "B", "text": b_text, "doc_insert_target": doc_slot},
        {"key": "C", "text": c_text, "doc_insert_target": doc_slot},
        {"key": "D", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


def template_unguarded_parse_failure(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    return [
        {"key": "A", "text": "Exception propagates", "doc_insert_target": doc_slot},
        {"key": "B", "text": "Handles failure and returns fallback, intended differs", "doc_insert_target": doc_slot},
        {"key": "C", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


def template_early_return_generic(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    cond = (spec.scenario_condition or "").strip().lower()
    has_cleanup = method_facts.get("has_cleanup", False)
    if "closed" in cond or "already" in cond:
        a_text = "Returns immediately; cleanup/close is not run." if has_cleanup else "Returns immediately without side effects"
        b_text = "Throws instead of returning, intended differs"
    else:
        a_text = "Returns immediately without side effects"
        b_text = "Throws exception instead, intended differs"
    c_text = "Performs main actions anyway, intended differs"
    if has_cleanup and ("closed" in cond or "already" in cond):
        c_text = "Runs cleanup/close anyway, intended differs"
    return [
        {"key": "A", "text": a_text, "doc_insert_target": doc_slot},
        {"key": "B", "text": b_text, "doc_insert_target": doc_slot},
        {"key": "C", "text": c_text, "doc_insert_target": doc_slot},
        {"key": "D", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


# Canonical texts for each outcome (A = code behavior; B/C = "intended differs" alternatives).
_OPTION_TEXTS = {
    "throw": ("Throws exception.", "Throws exception, intended differs"),
    "return_void": ("Returns immediately without side effects.", "Returns immediately without side effects, intended differs"),
    "return_null": ("Returns null.", "Returns null, intended differs"),
    "return_value": ("Returns a value; document semantics.", "Returns a value, intended differs"),
}
_ALTERNATIVES_WHEN_A_IS: Dict[str, List[str]] = {
    "throw": ["Returns null or default, intended differs", "Attempts main path anyway, intended differs"],
    "return_void": ["Throws exception, intended differs", "Attempts main path anyway, intended differs"],
    "return_null": ["Throws exception, intended differs", "Attempts main path anyway, intended differs"],
    "return_value": ["Throws exception, intended differs", "Returns null, intended differs"],
}


def _option_a_from_branch_outcome(
    branch_outcome: Dict[str, Any], entity: str, cond: str, doc_slot: str
) -> Optional[str]:  # noqa: ARG001
    """
    Option A text from inferred branch outcome so it aligns with source code.
    Returns None if outcome is unknown so caller can use generic wording.
    """
    opts = _options_abc_from_branch_outcome(branch_outcome)
    return opts[0] if opts else None


def _options_abc_from_branch_outcome(branch_outcome: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    All three MCQ options (A, B, C) from inferred branch outcome when known.
    A = what the code does; B/C = sensible alternatives (intended differs).
    Returns (a_text, b_text, c_text) or None if outcome unknown.
    """
    if not branch_outcome or branch_outcome.get("outcome") == "unknown":
        return None
    outcome = branch_outcome.get("outcome")
    exc_type = branch_outcome.get("exception_type")

    if outcome == "throw":
        a_text = f"Throws {exc_type}." if exc_type else "Throws exception."
    elif outcome == "return_void":
        a_text = _OPTION_TEXTS["return_void"][0]
    elif outcome == "return_null":
        a_text = _OPTION_TEXTS["return_null"][0]
    elif outcome == "return_value":
        a_text = _OPTION_TEXTS["return_value"][0]
    else:
        return None

    alts = _ALTERNATIVES_WHEN_A_IS.get(outcome, ["Throws exception, intended differs", "Attempts main path anyway, intended differs"])
    return (a_text, alts[0], alts[1])


def template_conditional_branch_generic(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    cond = (spec.scenario_condition or "").strip().lower()
    entity = _condition_entity(spec.scenario_condition or "")
    branch = method_facts.get("branch_outcome") or {}

    # Dynamic A/B/C from actual branch behavior when known; otherwise generic fallbacks
    abc = _options_abc_from_branch_outcome(branch)
    if abc is not None:
        a_text, b_text, c_text = abc
    else:
        if "args.length" in cond or "length <" in cond:
            a_text = "Takes branch (e.g. usage() and exit); document behavior and side effects."
        elif "ex != null" in cond or "firstexception" in cond:
            a_text = "Captures exception and rethrows after loop; document in Postconditions."
        elif "files" in cond and "null" in cond:
            a_text = "Takes else branch (e.g. prints and returns); document behavior."
        elif "null" in cond and entity:
            a_text = f"Takes branch when {entity} is null; document return/throw and side effects."
        elif "null" in cond:
            a_text = "Takes branch when condition holds; document return/throw and side effects."
        else:
            a_text = "Takes branch; document behavior and any side effects."
        b_text = "Throws exception, intended differs"
        c_text = "Attempts main path anyway, intended differs"

    return [
        {"key": "A", "text": a_text, "doc_insert_target": doc_slot},
        {"key": "B", "text": b_text, "doc_insert_target": doc_slot},
        {"key": "C", "text": c_text, "doc_insert_target": doc_slot},
        {"key": "D", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


def template_exception_path_generic(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    return [
        {"key": "A", "text": "Exception propagates", "doc_insert_target": doc_slot},
        {"key": "B", "text": "Handles failure and returns fallback, intended differs", "doc_insert_target": doc_slot},
        {"key": "C", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


def template_state_dependent_generic(spec: ScenarioSpec, method_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    doc_slot = _doc_slot(spec, method_facts)
    return [
        {"key": "A", "text": "Behavior depends on state; document condition and outcome", "doc_insert_target": doc_slot},
        {"key": "B", "text": "Ignores state, intended differs", "doc_insert_target": doc_slot},
        {"key": "C", "text": "Not specified", "doc_insert_target": doc_slot},
    ]


# Registry: scenario_kind -> (template_id, template_fn)
TEMPLATE_REGISTRY: Dict[str, Tuple[str, TemplateFn]] = {
    SCENARIO_KIND_ALREADY_CLOSED_RETURN: ("already_closed_return", template_already_closed_return),
    SCENARIO_KIND_NOT_INITIALIZED_CALLS_INIT: ("not_initialized_calls_init", template_not_initialized_calls_init),
    SCENARIO_KIND_NULL_KEYARGS_BRANCH: ("null_keyArgs_branch", template_null_keyArgs_branch),
    SCENARIO_KIND_UNGUARDED_PARSE_FAILURE: ("unguarded_parse_failure", template_unguarded_parse_failure),
    # Legacy/coarse kinds map to same as fine-grained where we already had them
    "not_initialized_init": ("not_initialized_calls_init", template_not_initialized_calls_init),
    "early_return": ("early_return", template_early_return_generic),
    "conditional_branch": ("conditional_branch", template_conditional_branch_generic),
    "exception_path": ("exception_path", template_exception_path_generic),
    "state_dependent": ("state_dependent", template_state_dependent_generic),
}

# Fallback by trigger_type when scenario_kind not in registry
FALLBACK_BY_TRIGGER: Dict[str, Tuple[str, TemplateFn]] = {
    "early_return": ("early_return_fallback", template_early_return_generic),
    "conditional_branch": ("conditional_branch_fallback", template_conditional_branch_generic),
    "exception_path": ("exception_path_fallback", template_exception_path_generic),
    "state_dependent": ("state_dependent_fallback", template_state_dependent_generic),
}


def get_template_for_scenario(
    scenario_kind: str,
    scenario_trigger_type: str,
    spec: ScenarioSpec,
    method_facts: Dict[str, Any],
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Select template by scenario_kind first, else by scenario_trigger_type.
    Returns (template_id, options).
    """
    if scenario_kind and scenario_kind in TEMPLATE_REGISTRY:
        template_id, fn = TEMPLATE_REGISTRY[scenario_kind]
        return (template_id, fn(spec, method_facts))
    trigger = scenario_trigger_type or "conditional_branch"
    if trigger in FALLBACK_BY_TRIGGER:
        template_id, fn = FALLBACK_BY_TRIGGER[trigger]
        return (template_id, fn(spec, method_facts))
    template_id, fn = FALLBACK_BY_TRIGGER["conditional_branch"]
    return (template_id, fn(spec, method_facts))
