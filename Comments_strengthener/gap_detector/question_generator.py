"""
Question Generation Module

Transforms detected gaps into clear, short questions for developers.

Supervisor alignment:
- MCQ first: when gap.suggested_options exist (including for scenario gaps), expose them.
  Only fall back to free-form when there are no options.
- Single scenario: one observable outcome per question (return behavior, exception, or side effect).
- Questions ≤20 words; shortening uses template rewrite, not truncation, so the key choice
  (throw vs return, entity name) is never cut off.
- Options: prefer ≤5 words; canonical compression for long option text; stable keys A/B/C
  for deterministic mapping in builder.py.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from gap_detector.models import Gap, Question, ScenarioSpec
from gap_detector.scenario_templates import (
    get_template_for_scenario,
    TEMPLATE_REGISTRY_VERSION,
)
from gap_detector.branch_outcome import infer_branch_outcome


def validate_scenario_question(
    gap: Gap, method_code: str, options: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Option-code consistency check for scenario questions. Returns either
    ("ok", options) or ("corrected", corrected_options) from template.
    """
    if getattr(gap, "type", None) != "execution_scenario_gap" or not options:
        return ("ok", options)

    trigger = getattr(gap, "scenario_trigger_type", None) or getattr(gap, "scenario_kind", None) or ""
    kind = getattr(gap, "scenario_kind", None) or ""
    cond = (getattr(gap, "scenario_condition", None) or "").strip().lower()
    code_lower = (method_code or "").lower()
    opts_text = " ".join(o.get("text", "") for o in options).lower()
    # Options should reference scenario_condition keywords (e.g. "closed", "initialized", "keyArgs")
    cond_tokens = [t for t in re.findall(r"\b[a-z_][a-z0-9_]*\b", cond) if len(t) > 2 and t not in ("true", "false", "entry", "at")]
    if cond_tokens and not any(t in opts_text for t in cond_tokens):
        gen = QuestionGenerator()
        return ("corrected", gen._scenario_template_options(gap, method_code))

    # Options must not claim that commit/cleanup runs when the method has no such operations.
    has_commit_in_code = "commit" in code_lower
    has_cleanup_in_code = "cleanup" in code_lower or "clean up" in code_lower
    opts_claim_cleanup_runs = (
        "cleanup runs" in opts_text or "runs cleanup" in opts_text
        or "cleanup anyway" in opts_text or "performs close" in opts_text
        or "close anyway" in opts_text
    )
    opts_claim_commit = "commit" in opts_text and "skips commit" not in opts_text
    if (opts_claim_commit and not has_commit_in_code) or (opts_claim_cleanup_runs and not has_cleanup_in_code):
        gen = QuestionGenerator()
        return ("corrected", gen._scenario_template_options(gap, method_code))
    # Options must not mention operations absent in method: warn/log (e.g. "Logs warning"), init call
    if ("logs " in opts_text or " warning" in opts_text or "warn" in opts_text) and "warn" not in code_lower and "log" not in code_lower:
        gen = QuestionGenerator()
        return ("corrected", gen._scenario_template_options(gap, method_code))
    if "init " in opts_text and "init()" not in code_lower and "init (" not in code_lower:
        gen = QuestionGenerator()
        return ("corrected", gen._scenario_template_options(gap, method_code))

    # Early return branch: no option may mention cleanup runs, commit attempted, or exception propagates
    if trigger == "early_return" or (cond in ("closed", "else branch") and "return" in (gap.evidence_snippet or "").lower()):
        for phrase in ("cleanup runs", "commit", "exception propagates", "performs close"):
            if phrase in opts_text:
                gen = QuestionGenerator()
                return ("corrected", gen._scenario_template_options(gap, method_code))
        return ("ok", options)

    # keyArgs == null: options must mention log warning and no commit, not "already closed"
    if "keyargs" in cond and "null" in cond:
        if "already closed" in opts_text and "log" not in opts_text:
            gen = QuestionGenerator()
            return ("corrected", gen._scenario_template_options(gap, method_code))
        return ("ok", options)

    # not_initialized_init: options must not be generic commit/cleanup (handled by impossible-tokens above)
    if kind == "not_initialized_init":
        if "commit" in opts_text or "cleanup" in opts_text:
            gen = QuestionGenerator()
            return ("corrected", gen._scenario_template_options(gap, method_code))
        return ("ok", options)

    # init fails: only code-consistent outcome is propagation unless there is a catch
    if "init" in cond and ("fail" in cond or "throw" in (gap.issue or "").lower()):
        has_catch = "catch" in code_lower
        if not has_catch and "fallback" in opts_text:
            if any("intended differs" not in o.get("text", "") for o in options if "fallback" in o.get("text", "").lower()):
                gen = QuestionGenerator()
                return ("corrected", gen._scenario_template_options(gap, method_code))
        return ("ok", options)

    return ("ok", options)


class QuestionGenerator:
    """Generates developer-friendly questions from gaps."""
    
    def __init__(self, question_budget: int = 2, scenario_budget: int = 1):
        """
        Initialize question generator.
        
        SUPERVISOR ALIGNMENT: Max 2 questions per method; simple, scenario-focused; short answers.
        
        Args:
            question_budget: Maximum number of questions to ask per method (default: 2)
            scenario_budget: Minimum number of scenario questions to reserve if scenarios exist (default: 1)
        """
        self.question_budget = question_budget
        self.scenario_budget = min(scenario_budget, question_budget)  # Can't exceed total budget
    
    def generate_questions(self, gaps: List[Gap], method_code: str) -> List[Question]:
        """
        Generate questions from gaps, respecting the question budget.
        Ensures at least one question is generated if gaps exist.
        
        Args:
            gaps: List of detected gaps
            method_code: Method source code for context
            
        Returns:
            List of questions, filtered by budget and priority
        """
        if not gaps:
            return []
        
        # Second line of defense: do not ask when code already decides behavior (deterministic exception).
        # Even if detector misclassified as ask, skip questions with deterministic parse evidence.
        gaps = self._drop_deterministic_exception_questions(gaps, method_code)
        if not gaps:
            return []
        
        # Filter gaps based on question budget
        filtered_gaps = self._filter_gaps_by_budget(gaps)
        
        # Ensure at least one question is generated if gaps exist
        # This guarantees every method with detected gaps gets at least one question
        if not filtered_gaps and gaps:
            # If budget filtering removed all gaps, take the highest priority one
            filtered_gaps = [gaps[0]]  # gaps are already sorted by priority
        
        questions = []
        for gap in filtered_gaps:
            questions.append(self._to_question(gap, method_code))
        
        return questions
    
    def _drop_deterministic_exception_questions(self, gaps: List[Gap], method_code: str) -> List[Gap]:
        """
        Second line of defense: do not ask when code already decides behavior.
        Drop (1) missing_implicit_exception and (2) execution_scenario_gap about invalid numeric
        whenever deterministic parse evidence is present (e.g. Integer.valueOf/parseInt unguarded).
        """
        if not method_code:
            return gaps
        has_deterministic_parse = bool(
            re.search(r'Integer\.(valueOf|parseInt)|Long\.(parseLong|valueOf)|Double\.(parseDouble|valueOf)', method_code)
        )
        if not has_deterministic_parse:
            return gaps

        def _is_invalid_numeric_scenario(gap: Gap) -> bool:
            """True if gap is about non-null invalid integer / parse outcome (code-determined)."""
            if getattr(gap, "type", None) != "execution_scenario_gap":
                return False
            q = (getattr(gap, "question", None) or "").lower()
            issue = (getattr(gap, "issue", None) or "").lower()
            return (
                ("invalid" in q and ("integer" in q or "number" in q or "parse" in q))
                or ("non-null" in q and ("invalid" in q or "valid integer" in q))
                or "not a valid integer" in q
                or "invalid integer" in issue or "invalid numeric" in issue
            )

        return [
            g for g in gaps
            if getattr(g, "type", None) != "missing_implicit_exception"
            and not (has_deterministic_parse and _is_invalid_numeric_scenario(g))
        ]
    
    def _filter_gaps_by_budget(self, gaps: List[Gap]) -> List[Gap]:
        """
        Filter gaps based on question budget.
        
        SUPERVISOR ALIGNMENT: Reserve minimum scenario budget to ensure scenario questions
        are not starved by structural gaps (preconditions, exceptions, return semantics).
        
        Rules:
        - Reserve scenario_budget for execution scenario gaps (if scenarios exist)
        - Remaining budget for structural gaps (preconditions, exceptions, return semantics, etc.)
        - Within each category, prioritize by gap priority (4-5 critical, 3 important, 2 nice to have, 1 completeness)
        - Remove duplicates using dedup_key
        """
        # Remove duplicates based on dedup_key
        seen_keys = set()
        unique_gaps = []
        for gap in gaps:
            if gap.dedup_key and gap.dedup_key in seen_keys:
                continue
            if gap.dedup_key:
                seen_keys.add(gap.dedup_key)
            unique_gaps.append(gap)
        
        # SUPERVISOR ALIGNMENT: Separate scenario gaps from structural gaps
        scenario_gaps = [g for g in unique_gaps if g.type == "execution_scenario_gap"]
        structural_gaps = [g for g in unique_gaps if g.type != "execution_scenario_gap"]
        
        selected = []
        
        # Step 1: Reserve scenario budget if scenarios exist
        if scenario_gaps:
            # SUPERVISOR ALIGNMENT: Sort scenario gaps by priority (higher first), then by stable tie-breaker
            # Use gap.id as tie-breaker for stable sorting (deterministic order)
            scenario_gaps_sorted = sorted(scenario_gaps, key=lambda g: (-g.priority, g.id))
            # Take up to scenario_budget scenario questions (most important first)
            selected.extend(scenario_gaps_sorted[:self.scenario_budget])
            remaining_budget = self.question_budget - len(selected)
        else:
            remaining_budget = self.question_budget
        
        # Step 2: Fill remaining budget with structural gaps
        if remaining_budget > 0 and structural_gaps:
            # Filter structural gaps by priority
            high_priority = [g for g in structural_gaps if g.priority >= 4]
            medium_priority = [g for g in structural_gaps if g.priority == 3]
            low_priority = [g for g in structural_gaps if g.priority == 2]
            completeness_priority = [g for g in structural_gaps if g.priority == 1]
            
            # Take high priority first, then medium, then low priority, then completeness
            if remaining_budget > 0:
                selected.extend(high_priority[:remaining_budget])
                remaining_budget = self.question_budget - len(selected)
            if remaining_budget > 0:
                selected.extend(medium_priority[:remaining_budget])
                remaining_budget = self.question_budget - len(selected)
            if remaining_budget > 0:
                selected.extend(low_priority[:remaining_budget])
                remaining_budget = self.question_budget - len(selected)
            if remaining_budget > 0:
                selected.extend(completeness_priority[:remaining_budget])
        
        return selected[:self.question_budget]
    
    def _to_question(self, gap: Gap, method_code: str) -> Question:
        """
        Convert a gap to a question object.
        
        Args:
            gap: The gap to convert
            method_code: Method source code for context
            
        Returns:
            Question object
        """
        # Scenario gaps: data-driven template from TEMPLATE_REGISTRY by scenario_kind, then fallback by trigger_type.
        if gap.type == "execution_scenario_gap" and gap.scenario_kind:
            question_text = self._generate_scenario_question_contract_slot(gap, method_code)
            question_text = self._shorten_question_by_rewrite(question_text)
            opts = self._scenario_template_options(gap, method_code)
            status, opts = validate_scenario_question(gap, method_code, opts)
            template_id = self.get_scenario_template_id(gap, method_code)
            return Question(
                id=gap.id,
                priority=gap.priority,
                category=gap.type,
                doc_slot=gap.doc_slot,
                question_text=question_text,
                context_code=self._extract_context(gap, method_code),
                options=self._normalize_options(opts),
                evidence_confidence=gap.evidence_confidence,
                fact_or_guarantee=gap.kind,
                scenario_kind=gap.scenario_kind,
                scenario_condition=gap.scenario_condition,
                scenario_template_id=template_id,
                template_registry_version=TEMPLATE_REGISTRY_VERSION,
            )
        elif (
            getattr(gap, "kind", None) == "limitation"
            or (gap.type or "").startswith("limitations_")
            or "limitation" in (gap.type or "").lower()
        ):
            # Limitations gaps: never use generic fallback; use gap.question or a concrete fallback.
            if gap.question:
                question_text = gap.question
            else:
                issue = (gap.issue or "this behavior").strip()
                question_text = f"What limitation should be documented about: {issue}?"
            question_text = self._shorten_question_by_rewrite(question_text)
            return Question(
                id=gap.id,
                priority=gap.priority,
                category=gap.type,
                doc_slot=gap.doc_slot,
                question_text=question_text,
                context_code=self._extract_context(gap, method_code),
                options=self._normalize_options(gap.suggested_options),
                evidence_confidence=gap.evidence_confidence,
                fact_or_guarantee=gap.kind
            )
        else:
            if gap.question:
                question_text = gap.question
            else:
                formatted = self._format_question_text(gap)
                if formatted is not None:
                    question_text = formatted
                else:
                    question_text = self._safe_fallback_question_text(gap)
        
        # Replace speculative "purged data" questions with safer stability question when no code evidence
        if gap.type == "return_semantics_guarantee":
            if "purged" in question_text.lower() or "purge" in question_text.lower():
                context_lower = (gap.evidence_snippet or "").lower()
                if "purge" not in context_lower and "truncat" not in context_lower and "rotate" not in context_lower:
                    question_text = "Does logFileSize stay constant after init()?"
        
        question_text = self._shorten_question_by_rewrite(question_text)
        
        return Question(
            id=gap.id,
            priority=gap.priority,
            category=gap.type,
            doc_slot=gap.doc_slot,
            question_text=question_text,
            context_code=self._extract_context(gap, method_code),
            options=self._normalize_options(gap.suggested_options),
            evidence_confidence=gap.evidence_confidence,
            fact_or_guarantee=gap.kind
        )
    
    def _shorten_question_by_rewrite(self, question_text: str) -> str:
        """
        Keep questions ≤20 words. Prefer template rewrite over truncation so the
        key choice (throw vs return, entity name) is never cut off.
        """
        if not question_text:
            return question_text
        
        words = question_text.split()
        if len(words) <= 20:
            return question_text
        
        # Extract condition and entity for rewrite templates (preserve one scenario).
        condition = self._extract_condition_for_rewrite(question_text)
        entity = self._extract_entity_for_rewrite(question_text)
        
        # Rewrite into a short pattern instead of truncating.
        q_lower = question_text.lower()
        if "throw" in q_lower and ("return" in q_lower or "normally" in q_lower):
            return f"If {condition}, does this method throw or return?" if condition else question_text
        if "throw" in q_lower or "exception" in q_lower:
            return f"If {condition}, which exception is thrown?" if condition else "Which exception does this method throw?"
        if "update" in q_lower or "modif" in q_lower or "field" in q_lower:
            return f"If {condition}, does it update {entity}?" if (condition and entity) else f"Does this method update {entity}?" if entity else question_text
        if "return" in q_lower:
            return f"If {condition}, what does this method return?" if condition else "What does this method return?"
        # Generic short form that keeps the condition
        if condition:
            return f"If {condition}, what should docs say?"
        return question_text
    
    def _extract_condition_for_rewrite(self, question_text: str) -> str:
        """Extract a short condition phrase for use in rewrite (e.g. 'X is null', 'method returns early')."""
        for pattern in [
            r'if\s+([^,?]+(?:,\s*[^,?]+)*)\s*,?\s*(?:\w+\s+)*\?',  # "If X, ...?"
            r'when\s+([^?]+)\?',
            r'when\s+([^.?]+)\.',
        ]:
            m = re.search(pattern, question_text, re.IGNORECASE)
            if m:
                cond = m.group(1).strip()
                # If the condition contains operators or important predicates, keep it intact
                operator_tokens = ["==", "!=", "<=", ">=", "<", ">", "=", "isEmpty", "contains"]
                if any(tok in cond for tok in operator_tokens):
                    return cond
                if len(cond.split()) <= 10:
                    return cond
                # Trim to first 8 words only when it does not contain operator-like tokens
                return ' '.join(cond.split()[:8])
        return "this condition"
    
    def _extract_entity_for_rewrite(self, question_text: str) -> str:
        """Extract a single entity (parameter, field name) for use in rewrite."""
        # e.g. "parameter X", "field foo", "var is null" -> X, foo, var
        m = re.search(r'(?:parameter|field|var)\s+(\w+)', question_text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r'(\w+)\s+(?:is null|changes)', question_text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Fallback: look for simple assignment-style hints in the text, e.g. "metadata = ..."
        m = re.search(r'(\w+)\s*=', question_text)
        if m:
            return m.group(1)
        return "field"
    
    def _truncate_condition_safely(self, condition: str, max_words: int = 10) -> str:
        """Truncate condition so we don't cut mid-expression (e.g. avoid ending with '==', '&&')."""
        cond_words = condition.split()
        if len(cond_words) <= max_words:
            return condition
        truncated = " ".join(cond_words[: max_words])
        # Drop trailing incomplete tokens
        for suffix in ("==", "!=", "&&", "||", "<", ">", "<=", ">="):
            if truncated.endswith(suffix) or truncated.rstrip().endswith(suffix):
                truncated = truncated.rstrip()[: -len(suffix)].rstrip()
                break
        return truncated

    def _generate_scenario_question_contract_slot(self, gap: Gap, method_code: str = None) -> str:
        """
        One observable outcome per question: return behavior, exception, or side effect.
        Question wording varies by condition type so not every question reads identically.
        """
        scenario_kind = gap.scenario_kind
        condition = (gap.scenario_condition or "").strip() or "this condition"
        condition = self._truncate_condition_safely(condition)
        cond_lower = condition.lower()

        if scenario_kind == "conditional_branch":
            method_code_lower = (method_code or "").lower() if method_code else ""
            context_code = (gap.evidence_snippet or "").lower()
            has_throws = "throws" in method_code_lower or "throws" in context_code
            if has_throws and ("null" in cond_lower or "== null" in cond_lower or "!= null" in cond_lower):
                null_match = re.search(r'(\w+)\s*(?:==|!=)\s*null', condition, re.IGNORECASE)
                var_name = null_match.group(1) if null_match else "param"
                return f"If {var_name} is null, does this method throw or return?"
            # Vary by condition type so questions are not all identical
            if "args.length" in cond_lower or "length <" in cond_lower or "args.length <" in condition:
                return f"If {condition}, does the method exit (e.g. usage) or continue?"
            if ("ex != null" in cond_lower or "firstexception" in cond_lower or
                "exception" in cond_lower and ("!=" in condition or "==" in condition)):
                return f"If an exception is captured in the loop, does the method rethrow it?"
            if "files" in cond_lower and "null" in cond_lower:
                return f"If files is null, what does the method do (e.g. print and return)?"
            # Alternate phrasing for variety
            return f"If {condition}, does this method take the branch and return, or throw?"

        if scenario_kind == "early_return":
            code_lower = (method_code or "").lower()
            has_cleanup = "cleanup" in code_lower or "clean up" in code_lower or "close" in code_lower
            if ("closed" in cond_lower or "already" in cond_lower) and has_cleanup:
                return f"If {condition}, does the method return without running cleanup?"
            return f"If {condition}, does the method return without performing the main path?"

        if scenario_kind == "not_initialized_init":
            return f"If not initialized, does this method call init and throw or return?"

        if scenario_kind == "state_dependent":
            field_match = re.search(r'field\s+(\w+)', gap.issue or "", re.I)
            field_name = field_match.group(1) if field_match else "the field"
            return f"If {condition}, does it update {field_name}?"

        if scenario_kind == "initialization":
            return f"If not initialized, does this method throw or call init?"

        # Default: one slot
        doc_slot = (gap.doc_slot or "").lower()
        if "exception" in doc_slot or "throws" in doc_slot:
            return f"If {condition}, which exception is thrown?"
        if "side" in doc_slot or "effect" in doc_slot:
            return f"If {condition}, does it update a field?"
        return f"If {condition}, what should docs say?"
    
    def _scenario_spec_from_gap(self, gap: Gap) -> ScenarioSpec:
        """Build ScenarioSpec from Gap for data-driven template selection."""
        kind = getattr(gap, "scenario_kind", None) or "conditional_branch"
        trigger = getattr(gap, "scenario_trigger_type", None) or kind
        condition = (getattr(gap, "scenario_condition", None) or "").strip()
        outcomes = getattr(gap, "scenario_outcomes", None) or []
        if not outcomes and getattr(gap, "scenario_expected_effects", None):
            outcomes = [s.strip() for s in gap.scenario_expected_effects.split(",") if s.strip()]
        return ScenarioSpec(
            scenario_id=gap.id,
            scenario_trigger_type=trigger,
            scenario_kind=kind,
            scenario_condition=condition,
            scenario_evidence=(getattr(gap, "scenario_evidence", None) or gap.evidence_snippet or ""),
            scenario_outcomes=outcomes,
            evidence_confidence=gap.evidence_confidence or "medium",
            code_determined=getattr(gap, "code_determined", None) or False,
        )

    def _method_facts_from_code(self, method_code: str, gap: Gap) -> Dict[str, Any]:
        """Extract method facts for template functions (e.g. doc_slot, has_commit, has_cleanup, branch_outcome)."""
        code_lower = (method_code or "").lower()
        facts = {
            "doc_slot": (gap.doc_slot or "Postconditions").strip(),
            "has_commit": "commit" in code_lower,
            "has_cleanup": "cleanup" in code_lower or "clean up" in code_lower,
            "has_warn": "warn" in code_lower or "log" in code_lower,
            "has_init_call": "init()" in code_lower or "init (" in code_lower,
        }
        if getattr(gap, "type", None) == "execution_scenario_gap":
            scenario_cond = getattr(gap, "scenario_condition", None) or ""
            evidence = getattr(gap, "scenario_evidence", None) or getattr(gap, "evidence_snippet", None) or ""
            facts["branch_outcome"] = infer_branch_outcome(method_code, scenario_cond, evidence)
        else:
            facts["branch_outcome"] = {"outcome": "unknown", "exception_type": None}
        return facts

    def _scenario_template_options(self, gap: Gap, method_code: str) -> List[Dict[str, str]]:
        """
        Data-driven: select template by scenario_kind from TEMPLATE_REGISTRY, else fallback by scenario_trigger_type.
        Returns options only (for validator and legacy callers). Use get_scenario_template_id for logging.
        """
        spec = self._scenario_spec_from_gap(gap)
        method_facts = self._method_facts_from_code(method_code, gap)
        _template_id, options = get_template_for_scenario(
            spec.scenario_kind, spec.scenario_trigger_type, spec, method_facts
        )
        return options

    def get_scenario_template_id(self, gap: Gap, method_code: str) -> str:
        """Return template_id for this scenario (for logging and reproducibility)."""
        spec = self._scenario_spec_from_gap(gap)
        method_facts = self._method_facts_from_code(method_code, gap)
        template_id, _ = get_template_for_scenario(
            spec.scenario_kind, spec.scenario_trigger_type, spec, method_facts
        )
        return template_id

    def _scenario_fallback_options(self, gap: Gap, question_text: str) -> List[Dict[str, str]]:
        """
        Fallback MCQ options for execution_scenario_gap when template not used (legacy path).
        Prefer _scenario_template_options for alignment.
        """
        return self._scenario_template_options(gap, "")
    
    def _is_concrete_issue(self, gap: Gap) -> bool:
        """True if gap has a concrete, specific issue (not generic). Used to decide when to use _format_question_text."""
        if not gap.issue or not gap.issue.strip():
            return False
        issue = gap.issue.strip()
        if len(issue) < 20:
            return False
        generic_phrases = ("no specific gaps", "general completeness", "completeness check", "nothing specific")
        if any(p in issue.lower() for p in generic_phrases):
            return False
        return True

    def _format_question_text(self, gap: Gap) -> Optional[str]:
        """
        Format question text from gap issue only when there is a concrete issue and no scenario template fits.
        SUPERVISOR: one scenario, short; return None if issue is not concrete (caller uses safe fallback).
        """
        if not self._is_concrete_issue(gap):
            return None
        issue = gap.issue.strip().rstrip(".")
        return f"Regarding: {issue}. How should it be documented?"

    def _safe_fallback_question_text(self, gap: Gap) -> str:
        """Fallback: one contract slot, concrete. Prefer doc_slot so detector-supplied question is used when possible."""
        doc_slot = (gap.doc_slot or "").strip().lower()
        if "precondition" in doc_slot:
            return "What precondition should docs specify here?"
        if "exception" in doc_slot or "throws" in doc_slot:
            return "What exception contract should docs specify here?"
        if "return" in doc_slot:
            return "What return-value contract should docs specify here?"
        if "side" in doc_slot or "effect" in doc_slot:
            return "What side-effect contract should docs specify here?"
        return "What contract should docs specify here?"
    
    def _extract_context(self, gap: Gap, method_code: str) -> str:
        """Extract relevant code context for the gap."""
        if gap.evidence_snippet:
            return gap.evidence_snippet
        
        # Fallback: extract method signature
        lines = method_code.split('\n')
        for line in lines[:5]:  # Check first 5 lines for method signature
            if '{' in line or '(' in line:
                return line.strip()
        
        return method_code[:200]  # Return first 200 chars as fallback
    
    def generate_question_json(self, gap: Gap, method_code: str, llm_client=None) -> Dict[str, Any]:
        """
        Generate question as JSON.
        
        Returns:
            Dict with question text and multiple‑choice `options` (if any were suggested on the gap).
        """
        question = self._to_question(gap, method_code)
        return {
            "question_text": question.question_text,
            "options": question.options,
            "suggested_best_option_confidence": question.evidence_confidence
        }

    # Preferred option length (words). Hard cap 20; prefer ≤5 for deterministic mapping.
    PREFERRED_OPTION_WORDS = 5
    MAX_OPTION_WORDS = 20

    # Canonical short labels for compression (maps meaning to stable short text for builder snippets).
    # Concurrency options MUST stay distinct: check most specific patterns first so we don't collapse
    # "thread-safe" / "not thread-safe" / "conditionally thread-safe" into one label.
    _OPTION_COMPRESS: List[Tuple[str, str]] = [
        # Concurrency: distinct short labels for assertion/bug-detection-friendly docs
        (r".*conditionally\s+thread[- ]?safe.*", "Conditionally thread-safe; document scope."),
        (r".*Method\s+must\s+be\s+thread[- ]?safe.*", "Thread-safe; document sync and visibility."),
        (r".*not\s+thread[- ]?safe.*synchronize\s+externally.*", "Not thread-safe; caller must synchronize."),
        (r".*Synchronization\s+is\s+internal.*(?:not\s+assume|no\s+guarantee).*", "No thread-safety guarantee; internal sync only."),
        (r".*synchronized\.\s+Callers\s+can\s+treat.*thread[- ]?safe.*", "Thread-safe for this object."),
        (r".*synchronized\s+but\s+state\s+interactions.*partially\s+safe.*", "Partially thread-safe; document scope."),
        # Generic concurrency fallback only for short phrases (avoids collapsing long distinct options)
        (r"\b(?:not\s+)?thread[- ]?safe\b", "Not thread-safe."),
        (r"\bthrow(?:s)?\s+(?:a\s+)?(?:NullPointerException|NPE)\b", "Throw NPE."),
        (r"\b(?:parameter|param)\s+(\w+)\s+must\s+not\s+be\s+null\b", r"Precondition \1 non-null."),
        (r"\b(?:parameter|param)\s+(\w+)\s+may\s+be\s+null\b", r"\1 nullable; method handles."),
        (r"\breturn(?:s)?\s+normally\b", "Return normally."),
        (r"\breturn(?:s)?\s+(?:early\s+)?without\s+(?:throwing|exception)\b", "Return without throwing."),
        (r"\b(?:document|add)\s+(?:as\s+)?(?:a\s+)?precondition\b", "Document precondition."),
        (r"\b(?:document|add)\s+(?:as\s+)?(?:an?\s+)?(?:@throws|exception)\b", "Document exception."),
        (r"\b(?:defensive\s+copy|copy\s+of\s+collection)\b", "Defensive copy."),
        (r"\b(?:live\s+view|view\s+of\s+internal)\b", "Live view."),
        (r"\b(?:synchronized|single\s+thread)\b", "Synchronized / single-thread."),
        (r"\b(?:update|modif(?:y|ies))\s+(?:the\s+)?(\w+)\s+field\b", r"Updates \1."),
        (r"\b(?:update|modif(?:y|ies))\s+fields?\b", "Updates field(s)."),
        # Cache: ban "cached indefinitely" (over-strong); use method-scoped safe wording
        (r".*cached\s+indefinitely.*", "No refresh guarantee. This method does not refresh the cached value after initialization unless the field is reset elsewhere."),
    ]

    # Ban over-strong cache wording; replace with method-scoped safe phrase (generalization-safe).
    _CACHED_INDEFINITELY_REPLACEMENT = "No refresh guarantee. This method does not refresh the cached value after initialization unless the field is reset elsewhere."

    def _normalize_options(self, options: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize options: preserve stable keys A/B/C; keep text short (prefer ≤5 words).
        Long options are compressed via canonical phrases; no truncation that drops the key differentiator.
        Ensures no two options have the same text (dedupe by keeping original short form for duplicates).
        Bans "cached indefinitely" globally; replaces with safe method-scoped wording.
        """
        if not options:
            return []
        
        normalized: List[Dict[str, str]] = []
        seen_texts: Dict[str, int] = {}  # text -> count of times we've seen it
        for opt in options:
            if not isinstance(opt, dict):
                continue
            text = opt.get("text", "")
            if "cached indefinitely" in text.lower():
                text = self._CACHED_INDEFINITELY_REPLACEMENT
            key = opt.get("key") or opt.get("id")
            if not text or not text.strip():
                continue
            compressed = self._compress_option_text(text.strip())
            # If this compressed text was already used, keep a short distinct form from the original
            if compressed in seen_texts:
                seen_texts[compressed] += 1
                # Use first N words of original to keep distinct; cap at MAX_OPTION_WORDS
                words = text.strip().split()
                fallback = " ".join(words[: min(len(words), self.MAX_OPTION_WORDS)])
                if fallback != compressed:
                    compressed = fallback
            else:
                seen_texts[compressed] = 1
            normalized.append({"key": str(key) if key is not None else "", "text": compressed})
        return normalized

    def _compress_option_text(self, text: str) -> str:
        """
        Prefer ≤5 words; hard cap 20. Use canonical short phrases when possible;
        otherwise trim only if still over 20 words (avoid cutting the key differentiator).
        """
        for pattern, replacement in self._OPTION_COMPRESS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return m.expand(replacement)
                except re.error:
                    return replacement
        words = text.split()
        if len(words) <= self.PREFERRED_OPTION_WORDS:
            return text
        if len(words) <= self.MAX_OPTION_WORDS:
            return text
        return " ".join(words[: self.MAX_OPTION_WORDS])

    def _shorten_question_if_needed(self, question_text: str) -> str:
        """Backward compatibility: delegates to template-based rewrite (_shorten_question_by_rewrite)."""
        return self._shorten_question_by_rewrite(question_text)

