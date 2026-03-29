"""
Infer what a conditional branch does (return vs throw) from source code.

Used to align scenario question option A with actual behavior so templates
don't show generic "Takes branch; document..." when the code clearly returns
null, returns void, or throws. Improves generalization across diverse methods.
"""

import re
from typing import Any, Dict


def infer_branch_outcome(
    method_code: str,
    scenario_condition: str,
    evidence_snippet: str = "",
) -> Dict[str, Any]:
    """
    Infer what happens when the scenario condition is true: return (void/null/value) or throw.

    Returns dict with:
      - outcome: "return_void" | "return_null" | "return_value" | "throw" | "unknown"
      - exception_type: str or None (e.g. "FileNotFoundException")
    """
    if not (scenario_condition or evidence_snippet or method_code):
        return {"outcome": "unknown", "exception_type": None}

    # Prefer evidence snippet (contains the if and a few following lines)
    block = (evidence_snippet or "").strip() or (method_code or "").strip()
    if not block and method_code and scenario_condition:
        block = _extract_block_for_condition(method_code, scenario_condition)

    if not block:
        return {"outcome": "unknown", "exception_type": None}

    block_lower = block.lower()
    # Normalize: remove line comments and collapse whitespace for simpler matching
    block_clean = re.sub(r'//[^\n]*', '', block)
    block_clean = re.sub(r'/\*.*?\*/', '', block_clean, flags=re.DOTALL)

    # Order matters: throw is often first in guard clauses
    if re.search(r'\bthrow\s+new\s+(\w+)', block_clean):
        exc_match = re.search(r'\bthrow\s+new\s+(\w+)', block_clean)
        exc_type = exc_match.group(1) if exc_match else None
        return {"outcome": "throw", "exception_type": exc_type}

    if re.search(r'\breturn\s*;', block_clean):
        return {"outcome": "return_void", "exception_type": None}
    if re.search(r'\breturn\s+null\s*;', block_clean):
        return {"outcome": "return_null", "exception_type": None}
    if re.search(r'\breturn\s+[^;]+;', block_clean):
        return {"outcome": "return_value", "exception_type": None}

    return {"outcome": "unknown", "exception_type": None}


def _extract_block_for_condition(method_code: str, scenario_condition: str) -> str:
    """
    Find an if-block in method_code that matches the scenario condition and return its body.
    Uses flexible matching so "columnName is null" matches "columnName == null" in code.
    """
    if not scenario_condition or not method_code:
        return ""

    scenario_lower = scenario_condition.strip().lower()
    # Key tokens from scenario (e.g. "columnname", "null", "line", "nummaps")
    scenario_tokens = set(re.findall(r"[a-z0-9_]+", scenario_lower.replace(" is ", " ")))

    for m in re.finditer(r"if\s*\([^)]+\)\s*\{", method_code):
        start = m.end()
        if_head = method_code[m.start() : m.end()]
        code_cond = re.search(r"if\s*\(\s*([^)]+)\s*\)", if_head)
        if not code_cond:
            continue
        code_cond_str = code_cond.group(1).strip().lower()
        code_tokens = set(re.findall(r"[a-z0-9_]+", code_cond_str))
        overlap = code_tokens & scenario_tokens
        if not overlap:
            continue
        # Require at least one substantive token (not just "null" or "true")
        substantive = overlap - {"null", "true", "false", "is", "not", "and", "or"}
        if not substantive and "null" not in overlap:
            continue
        if len(overlap) >= 2 or "null" in overlap or substantive:
            depth = 1
            i = start
            while i < len(method_code) and depth > 0:
                if method_code[i] == "{":
                    depth += 1
                elif method_code[i] == "}":
                    depth -= 1
                i += 1
            body = method_code[start : i - 1] if depth == 0 else method_code[start : start + 600]
            return body
    return ""
