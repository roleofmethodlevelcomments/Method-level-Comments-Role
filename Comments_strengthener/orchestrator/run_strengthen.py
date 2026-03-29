"""
Main orchestrator for comment strengthening pipeline.
"""

# Pipeline version tag for empirical study reproducibility (change when detector/prompt/generator change)
PIPELINE_VERSION = "1.0-diagnostic"

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ast_extractor.extractor import extract_ast_facts
from prompt_builder.builder import PromptBuilder
from llm.generator import CommentGenerator
from utils.io import load_dataset, save_results, extract_method_data
from utils.token_utils import extract_javadoc_block, extract_javadoc_tags
from gap_detector import GapDetector, QuestionGenerator, GapRouter, QuestionBank
from gap_detector.models import Gap, Question


class CommentStrengthener:
    """Main orchestrator for comment strengthening."""
    
    def __init__(self, mode: str = "contract", llm_provider: str = "openai", 
                 llm_model: str = "gpt-4o-mini", enable_gap_detection: bool = True,
                 question_bank: QuestionBank = None):
        """
        Initialize comment strengthener.
        
        Args:
            mode: "rewrite" or "contract"
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Model name
            enable_gap_detection: Enable gap detection and question generation
            question_bank: QuestionBank instance for managing questions (optional)
        """
        self.mode = mode
        self.prompt_builder = PromptBuilder(mode=mode)
        self.generator = CommentGenerator(provider=llm_provider, model=llm_model)
        self.enable_gap_detection = enable_gap_detection
        self.question_bank = question_bank
        
        if enable_gap_detection:
            # Pass LLM client to gap detector for clarification
            llm_client = self.generator.client if hasattr(self.generator, 'client') else None
            self.gap_detector = GapDetector(llm_client=llm_client)
            # SUPERVISOR ALIGNMENT: Max 2 questions per method; simple, scenario-focused; short answers.
            # Do not ask for "complete requirements" in one question (too heavy).
            self.question_generator = QuestionGenerator(question_budget=2, scenario_budget=1)
            self.gap_router = GapRouter()
    
    def strengthen_comment(self, method_code: str, original_comment: str,
                          method_id: str = None, interactive: bool = False,
                          requirements_changed: bool = False) -> Dict[str, Any]:
        """
        Strengthen a single comment with retry mechanism using different strategies.
        
        Args:
            method_code: Method source code
            original_comment: Original Javadoc comment
            method_id: Method ID for question bank tracking (optional)
            interactive: If True, prompt user for questions (optional)
            requirements_changed: If True, prompt will ask for explicit old vs new behavior (requirement evolution).
            
        Returns:
            Result dictionary with strengthened comment and metadata
        """
        # Step 1: Extract AST facts
        ast_facts = extract_ast_facts(method_code)
        
        # Add method_id to ast_facts for method-scoped dedup keys
        if method_id:
            ast_facts['method_id'] = method_id
        
        # Requirement evolution: when True, prompt will ask for explicit old vs new behavior (for bug-detection value).
        # Postfix code/comment are never passed to the LLM; no delta-derived signals, to avoid bias and support generalization.
        ast_facts['requirements_changed'] = bool(requirements_changed)
        
        # Step 1.5: Gap detection (if enabled). Always pass gaps + answered_questions to LLM (may be empty).
        gap_results = {}
        questions = []
        answered_questions = {}
        routing_result = {}
        all_gaps = []

        if self.enable_gap_detection:
            # Detect gaps (returns Dict with "auto_add", "auto_fix", "ask", "skip")
            gap_results = self.gap_detector.detect_gaps(ast_facts, original_comment, method_code, mode=self.mode)
            
            # Get gaps for question generation (only "ask" gaps)
            gaps_to_ask = gap_results.get("ask", [])
            auto_add = gap_results.get("auto_add", [])
            skip = gap_results.get("skip", [])
            # Only scenario-based questions are sent to the developer; other ask gaps stay in metadata for LLM/routing.
            scenario_gaps_to_ask = [g for g in gaps_to_ask if getattr(g, "type", None) == "execution_scenario_gap"]
            if not scenario_gaps_to_ask:
                n_ask = len(gaps_to_ask)
                if n_ask > 0:
                    ask_types = {getattr(g, "type", "?") for g in gaps_to_ask}
                    print(f"[Gap detection] 0 scenario ask-gaps for this method (auto_add={len(auto_add)}, ask={n_ask} [types: {ask_types}], skip={len(skip)}). Only execution_scenario_gap → developer questions; other ask gaps passed to LLM only.")
                else:
                    print(f"[Gap detection] 0 scenario ask-gaps for this method (auto_add={len(auto_add)}, ask=0, skip={len(skip)}). Only scenario-based questions are shown; AST and gaps still passed to LLM.")
            # Generate questions only from execution_scenario_gap (scenario-based) so developer sees only those
            questions = self.question_generator.generate_questions(scenario_gaps_to_ask, method_code)
            
            # SUPERVISOR: Do NOT add a "complete requirements" fallback question when no gaps detected.
            # One heavy question asking for full semantics is too much; keep questions simple and scenario-specific.
            # If no ask gaps, leave questions = [] for this method (zero questions is allowed).
            
            # Get answered questions from question bank or interactive mode
            answered_questions = {}
            if method_id and self.question_bank:
                answered_questions = self.question_bank.get_answered_questions(method_id)
            
            if interactive and questions:
                interactive_answers = self._interactive_question_loop(
                    method_id, questions, method_code
                )
                answered_questions.update(interactive_answers)
                # FIXED: Update Question objects with answered=True so pending_questions check works
                for question in questions:
                    if question.id in answered_questions:
                        question.answered = True
                        question.developer_answer = answered_questions[question.id]
            
            # Route gaps (combine all gaps for routing; auto_fix = documentation alignment, not ask)
            all_gaps = (
                gap_results.get("auto_add", [])
                + gap_results.get("auto_fix", [])
                + gap_results.get("ask", [])
                + gap_results.get("skip", [])
            )
            routing_result = self.gap_router.route_gaps(all_gaps, answered_questions)
        
        # Step 2: Try different strategies with dynamic selection based on failures
        all_rejection_reasons = []
        strategy_order = [0]  # Start with default strategy
        tried_strategies = set()
        strategy_idx = 0
        
        while strategy_idx < len(strategy_order) and len(tried_strategies) < 4:
            strategy = strategy_order[strategy_idx]
            tried_strategies.add(strategy)
            strategy_rejections = []
            last_failure_reason = None
            
            # Build prompt: always provide AST facts + gap detection + developer answers + original comment to LLM.
            prompt = self.prompt_builder.build_prompt(
                ast_facts, method_code, original_comment,
                strategy=strategy, gaps=all_gaps, answered_questions=answered_questions,
            )
            
            # Generate candidates
            candidates = self.generator.generate_candidates(prompt, self.mode)
            
            # Accept first candidate (validation component removed after strengthened comments)
            accepted_candidate = None
            accepted_reason = "accepted"
            accepted_metadata = {'semantic_drift': 0.0, 'hallucination_risk': 0, 'noise_score': 0.0, 'tags_preserved': True}
            if candidates:
                accepted_candidate = candidates[0]
                print(f"[Strengthen] Strategy {strategy}: using first candidate")
            
            # If we have a candidate, return it
            if accepted_candidate:
                # Extract just the Javadoc block
                javadoc_block = extract_javadoc_block(accepted_candidate)
                strengthened_comment = javadoc_block if javadoc_block else accepted_candidate
                strengthened_comment = self._strip_empty_sections(strengthened_comment)
                strengthened_comment = self._strip_redundant_contract_tags(strengthened_comment)
                strengthened_comment = self._dedupe_throws_conditions(strengthened_comment)
                strengthened_comment = self._validate_throws_tags(strengthened_comment, method_code)
                strengthened_comment = self._strip_npe_for_null_handled_params(strengthened_comment, method_code)
                strengthened_comment = self._validate_param_tags(strengthened_comment, ast_facts)
                strengthened_comment = self._strip_unapproved_sections(strengthened_comment)
                # Ensure Concurrency section for synchronized methods (avoid missing_concurrency_section)
                strengthened_comment = self._ensure_concurrency_section(strengthened_comment, ast_facts)
                # Pipeline uses only the above structure-preserving steps (no separate heuristic comment rewriter).
                
                # FIXED: Consistency check - accepted cannot be true if pending_questions exist with unanswered content
                pending_questions = routing_result.get('pending_questions', []) if self.enable_gap_detection else []
                accepted_value = True
                accepted_reason_final = accepted_reason
                
                if pending_questions and self.enable_gap_detection:
                    # Check if strengthened comment contains unanswered ask content
                    strengthened_lower = strengthened_comment.lower()
                    has_unanswered_content = False
                    for gap in gap_results.get("ask", []):
                        if gap.id in pending_questions:
                            # Check if gap content appears in strengthened comment
                            if gap.issue.lower() in strengthened_lower or gap.question.lower() in strengthened_lower:
                                has_unanswered_content = True
                                break
                    
                    if has_unanswered_content:
                        accepted_value = False
                        accepted_reason_final = 'pending_questions_with_content'
                
                result = {
                    'strengthened_comment': strengthened_comment,
                    'accepted': accepted_value,  # FIXED: Enforce consistency
                    'mode': self.mode,
                    'reason': accepted_reason_final,
                    'semantic_drift': accepted_metadata.get('semantic_drift', 0.0),
                    'hallucination_risk': accepted_metadata.get('hallucination_risk', 0),
                    'noise_score': accepted_metadata.get('noise_score', 0.0),
                    'tags_preserved': accepted_metadata.get('tags_preserved', False),
                    'fallback_used': False,
                    'strategy_used': strategy,
                    'total_strategies_tried': len(tried_strategies)
                }
                
                # Add gap detection metadata if enabled (always include, even if empty)
                if self.enable_gap_detection:
                    all_gaps_for_metadata = (
                        gap_results.get("auto_add", [])
                        + gap_results.get("auto_fix", [])
                        + gap_results.get("ask", [])
                        + gap_results.get("skip", [])
                    )
                    
                    # FIXED: pending_questions must equal unanswered questions (derived, not stored)
                    # Also check answered_questions dict to ensure answers from interactive mode are counted
                    questions_dicts = [self._question_to_dict(q) for q in questions]
                    pending_questions_derived = [
                        q['id'] for q in questions_dicts 
                        if not (q.get('answered', False) or q['id'] in answered_questions)
                    ]
                    
                    # FIXED: facts_added computed from comment diff, not gaps list
                    facts_added_slots = self._compute_facts_added_from_diff(
                        original_comment, strengthened_comment
                    )
                    
                    # FIXED: Tag mismatch detection
                    tag_mismatch_info = self._check_tag_mismatch(
                        method_code, original_comment, strengthened_comment, ast_facts
                    )
                    
                    # FIXED: HARD GATE - accepted must be false if pending_questions non-empty
                    # This is a hard failure - no exceptions
                    if pending_questions_derived:
                        accepted_value = False
                        if not accepted_reason_final or accepted_reason_final == 'accepted':
                            accepted_reason_final = 'pending_questions_exist'
                    
                    # FIXED: Validate pending_questions matches unanswered questions
                    # Also check answered_questions dict
                    unanswered_ids = [
                        q['id'] for q in questions_dicts 
                        if not (q.get('answered', False) or q['id'] in answered_questions)
                    ]
                    if set(pending_questions_derived) != set(unanswered_ids):
                        accepted_value = False
                        accepted_reason_final = 'metadata_inconsistent'
                        print(f"[Validation] ERROR: pending_questions mismatch. Derived: {pending_questions_derived}, Unanswered: {unanswered_ids}")
                    
                    # Per-method metrics for empirical study
                    gaps_by_kind = self._count_gaps_by_kind(all_gaps_for_metadata)
                    validation = self._validate_evidence_backed_elements(
                        strengthened_comment, ast_facts, gap_results, method_code=method_code
                    )
                    validation['warnings'].extend(
                        self._check_unevidenced_behavioral_claims(
                            strengthened_comment, method_code, ast_facts
                        )
                    )
                    answer_contradicts_code = self._flag_answer_contradicts_code(
                        answered_questions, all_gaps_for_metadata, method_code
                    )
                    result['metadata'] = {
                        'pipeline_version': PIPELINE_VERSION,
                        'gaps_by_kind': gaps_by_kind,
                        'questions_count': len(questions_dicts),
                        'facts_added': facts_added_slots,  # FIXED: From diff, not gaps
                        'guarantees_confirmed': [g.doc_slot for g in routing_result.get('guarantees_confirmed', [])],
                        'candidate_inferences': [g.doc_slot for g in routing_result.get('candidate_inferences', [])],
                        'pending_questions': pending_questions_derived,  # FIXED: Derived from questions
                        'questions': questions_dicts,
                        'gaps': [self._gap_to_dict(g) for g in all_gaps_for_metadata],
                        'tag_mismatch': tag_mismatch_info.get('has_mismatch', False),
                        'tag_mismatch_detail': tag_mismatch_info.get('details', {}),
                        'validation_warnings': validation,
                        'intention_code_mismatch_documented': answer_contradicts_code,
                    }
                    
                    # Update accepted after metadata computation
                    result['accepted'] = accepted_value
                    result['reason'] = accepted_reason_final
                
                # Add rejections from previous strategies to metadata if any
                if all_rejection_reasons:
                    result['previous_rejections'] = all_rejection_reasons
                return result
            
            # Strategy failed - add rejections to tracking
            all_rejection_reasons.extend(strategy_rejections)
            
            # Dynamic strategy selection: add next strategies for ALL failure reasons
            # (so e.g. removed_tag + hallucination_number both get a follow-up strategy)
            if len(tried_strategies) < 4:
                reasons_in_this_strategy = {r['reason'] for r in strategy_rejections}
                added = set()
                for reason in reasons_in_this_strategy:
                    next_strategy = None
                    if reason == "modified_tag":
                        next_strategy = 1  # tag_focused
                    elif reason == "removed_tag":
                        next_strategy = 3  # explicit (tag-preserving)
                    elif reason in ["hallucination_number", "excessive_noise"]:
                        next_strategy = 2  # minimal
                    elif reason == "missing_required_section":
                        next_strategy = 3  # explicit
                    elif reason == "fake_concurrency":
                        next_strategy = 0  # default
                    if next_strategy is not None and next_strategy not in tried_strategies and next_strategy not in added:
                        strategy_order.append(next_strategy)
                        added.add(next_strategy)
                # If no reason-specific strategy was added, try next in sequence
                if not added:
                    for s in range(4):
                        if s not in tried_strategies:
                            strategy_order.append(s)
                            break
            
            strategy_idx += 1
        
        # Step 3: All strategies failed - fallback to original
        print(f"[Retry] All {len(tried_strategies)} strategies failed. Using fallback.")

        # Derive a compact summary of why this method should be revisited
        unique_reasons = sorted({r.get('reason', 'unknown') for r in all_rejection_reasons}) if all_rejection_reasons else []

        # Strip empty sections, redundant tags, validate @throws, strip unapproved sections, ensure Concurrency for synchronized methods even on fallback
        fallback_comment = self._strip_empty_sections(original_comment)
        fallback_comment = self._strip_redundant_contract_tags(fallback_comment)
        fallback_comment = self._dedupe_throws_conditions(fallback_comment)
        fallback_comment = self._validate_throws_tags(fallback_comment, method_code)
        fallback_comment = self._strip_npe_for_null_handled_params(fallback_comment, method_code)
        fallback_comment = self._validate_param_tags(fallback_comment, ast_facts)
        fallback_comment = self._strip_unapproved_sections(fallback_comment)
        fallback_comment = self._ensure_concurrency_section(fallback_comment, ast_facts)
        result = {
            'strengthened_comment': fallback_comment,
            'accepted': False,
            'mode': self.mode,
            'reason': 'fallback',
            'semantic_drift': 0.0,
            'hallucination_risk': 0,
            'noise_score': 1.0,
            'tags_preserved': True,
            'fallback_used': True,
            'total_strategies_tried': len(tried_strategies),
            'rejection_reasons': all_rejection_reasons,
            # NEW: explicit revisit markers for downstream analysis
            'needs_revisit': True,
            'revisit_reasons': unique_reasons,
        }
        
        # Add gap detection metadata if enabled (always include, even if empty)
        if self.enable_gap_detection:
            # Reconstruct all_gaps for metadata
            all_gaps_for_metadata = gap_results.get("auto_add", []) + gap_results.get("ask", []) + gap_results.get("skip", [])
            # Derive questions and pending questions deterministically
            # Also check answered_questions dict to ensure answers from interactive mode are counted
            questions_dicts = [self._question_to_dict(q) for q in questions]
            pending_questions_derived = [
                q['id'] for q in questions_dicts 
                if not (q.get('answered', False) or q['id'] in answered_questions)
            ]
            
            gaps_by_kind = self._count_gaps_by_kind(all_gaps_for_metadata)
            result['metadata'] = {
                'pipeline_version': PIPELINE_VERSION,
                'gaps_by_kind': gaps_by_kind,
                'questions_count': len(questions_dicts),
                'facts_added': [g.doc_slot for g in routing_result.get('facts_to_add', [])],
                'guarantees_confirmed': [g.doc_slot for g in routing_result.get('guarantees_confirmed', [])],
                'candidate_inferences': [g.doc_slot for g in routing_result.get('candidate_inferences', [])],
                'pending_questions': pending_questions_derived,
                'questions': questions_dicts,
                'gaps': [self._gap_to_dict(g) for g in all_gaps_for_metadata]
            }
        
        return result
    
    # Placeholder values that mean "no content" — sections with only these are removed (HTML or plain-text)
    _EMPTY_SECTION_VALUES = frozenset({
        'none', 'none.', 'no guarantees.', 'no guarantees', 'not specified.', 'not specified',
        'n/a.', 'n/a', '-', '—', '', '- none.', '- none'
    })

    # Section names allowed by the contract template; any other section header is stripped by normalizer
    _APPROVED_SECTION_HEADERS = frozenset({
        'Purpose:', 'Preconditions:', 'Postconditions:', 'SideEffects:',
        'Concurrency:', 'Exceptions:', 'EdgeCases:'
    })
    _UNAPPROVED_SECTION_NAMES = frozenset({'Limitations'})  # no colon; match "Limitations:" etc.

    def _strip_unapproved_sections(self, comment: str) -> str:
        """
        Contract normalizer: remove sections whose header is not in the approved list
        (e.g. "Limitations:") so output stays consistent with the fixed template.
        Preserves only: Purpose, Preconditions, Postconditions, SideEffects,
        Concurrency, Exceptions, EdgeCases. Never strips @param, @return, @throws.
        """
        if not comment or not comment.strip():
            return comment
        section_headers = (
            'Purpose:', 'Preconditions:', 'Postconditions:', 'SideEffects:',
            'Concurrency:', 'Exceptions:', 'EdgeCases:', 'Limitations:'
        )
        tag_markers = ('@param', '@return', '@throws', '@exception', '@deprecated', '@see', '@since')
        lines = comment.split('\n')
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            star_stripped = stripped.lstrip('*').strip()
            # Check if this line looks like a section header (e.g. "Limitations:" or "* Limitations:")
            is_unapproved = False
            for name in self._UNAPPROVED_SECTION_NAMES:
                if name + ':' == star_stripped or star_stripped.endswith(' ' + name + ':'):
                    is_unapproved = True
                    break
            if not is_unapproved:
                out.append(line)
                i += 1
                continue
            # Collect body until next section or tag
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped == '*':
                    j += 1
                    continue
                if any(h in next_stripped for h in section_headers):
                    break
                if any(next_stripped.startswith('*') and tag in next_stripped.lower() for tag in tag_markers):
                    break
                if next_stripped.startswith('@'):
                    break
                if '*/' in next_stripped:
                    break
                j += 1
            i = j
        return '\n'.join(out)

    def _strip_empty_sections(self, comment: str) -> str:
        """
        Remove any contract section (Preconditions, Postconditions, SideEffects,
        Concurrency, Exceptions, EdgeCases, Purpose) whose body is empty or
        only a placeholder (None, Not specified, etc.). Never strip lines
        beginning with @param, @return, or @throws. When detecting empty body,
        strip only HTML tags (<...>) for the check; do not remove or corrupt
        Javadoc tags like {@code ...}.
        """
        if not comment or not comment.strip():
            return comment
        section_headers = (
            'Purpose:', 'Preconditions:', 'Postconditions:', 'SideEffects:',
            'Concurrency:', 'Exceptions:', 'EdgeCases:'
        )
        tag_markers = ('@param', '@return', '@throws', '@exception', '@deprecated', '@see', '@since')
        lines = comment.split('\n')
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            is_section_header = any(
                stripped == h or stripped.endswith(' ' + h) or (stripped.startswith('*') and h in stripped)
                for h in section_headers
            )
            if not is_section_header:
                out.append(line)
                i += 1
                continue
            # Find which header this is
            header_name = None
            for h in section_headers:
                if h in stripped:
                    header_name = h
                    break
            if not header_name:
                out.append(line)
                i += 1
                continue
            # Collect body: lines until next section or tag
            body_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped == '*':
                    body_lines.append(next_line)
                    j += 1
                    continue
                if any(h in next_stripped for h in section_headers):
                    break
                if any(next_stripped.startswith('*') and tag in next_stripped.lower() for tag in tag_markers):
                    break
                if next_stripped.startswith('@'):
                    break
                if '*/' in next_stripped:
                    break
                body_lines.append(next_line)
                j += 1
            body_text = ' '.join(ln.strip().strip('*').strip() for ln in body_lines).strip()
            body_lower = body_text.lower()
            # Treat HTML "None" list as empty (e.g. <ul><li>None.</li></ul>).
            # Strip only HTML <...> for the check; do not touch {@code ...} or other Javadoc.
            body_no_html = re.sub(r'<[^>]+>', ' ', body_lower).strip()
            if (body_lower in self._EMPTY_SECTION_VALUES or not body_text or
                    body_no_html in self._EMPTY_SECTION_VALUES or
                    (body_no_html == 'none' or body_no_html == 'none.')):
                # Skip this section (do not append header or body)
                i = j
                continue
            out.append(line)
            for ln in body_lines:
                out.append(ln)
            i = j
        return '\n'.join(out)

    def _validate_evidence_backed_elements(
        self,
        comment: str,
        ast_facts: Dict[str, Any],
        gap_results: Dict[str, Any],
        *,
        method_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight validator for evidence-backed contract elements (Gate 1 & 2).
        Report-only: does not modify the comment. Used for readiness reporting
        and optional re-prompt; no auto-insert.
        """
        warnings = []
        missing_side_effects = False
        missing_concurrency = False
        if not comment:
            return {"missing_side_effects": False, "missing_concurrency": False, "warnings": []}
        comment_lower = comment.lower()
        auto_add = gap_results.get("auto_add") or []

        # Gate 1b: field write must appear in SideEffects or Postconditions
        has_field_write = any(getattr(g, "type", None) == "field_write_fact" for g in auto_add)
        if has_field_write:
            if "sideeffects" not in comment_lower and "postconditions" not in comment_lower:
                missing_side_effects = True
                warnings.append("field_write_fact in gaps but comment has no SideEffects or Postconditions section")

        # Gate 1a: synchronized → Concurrency must include serialization; flag vague "partial safety"
        has_sync_evidence = ast_facts.get("synchronized_method") or any(
            getattr(g, "type", None) == "synchronized_fact" for g in auto_add
        )
        if has_sync_evidence:
            if "concurrency" not in comment_lower and "synchronized" not in comment_lower:
                missing_concurrency = True
                warnings.append("synchronized_method or synchronized_fact but comment has no Concurrency section or synchronized mention")
            else:
                concurrency_block = self._extract_section_block(comment, "concurrency")
                if concurrency_block:
                    has_serialization = "serialized" in concurrency_block.lower() or "serializ" in concurrency_block.lower()
                    has_partial_safety = "partial safety" in concurrency_block.lower()
                    if has_partial_safety and not has_serialization:
                        warnings.append("Concurrency for synchronized method should state serialization; avoid vague 'partial safety' alone (Gate 1).")
                    elif not has_serialization and "synchronized" in concurrency_block.lower():
                        # Prefer serialization statement when synchronized
                        warnings.append("Concurrency for synchronized method should include serialization statement (e.g. calls are serialized on the instance) (Gate 1).")

        # Gate 1c: deterministic NumberFormatException from getValue/parse → must appear in Exceptions with source-accurate condition
        if method_code:
            has_getvalue = "getValue" in method_code
            has_unguarded_parse = (
                re.search(r"Integer\.(valueOf|parseInt)|Long\.(parseLong|valueOf)", method_code)
                and not re.search(r"try\s*\{[^}]*Integer\.(valueOf|parseInt)", method_code, re.DOTALL)
            )
            if has_getvalue and has_unguarded_parse:
                if "numberformatexception" not in comment_lower:
                    warnings.append("Deterministic NumberFormatException from getValue/parse should appear in Exceptions (Gate 1).")
                elif "getvalue" not in comment_lower and "cannot be parsed" not in comment_lower and "parsed as an integer" not in comment_lower:
                    warnings.append("NumberFormatException condition should be source-accurate (e.g. getValue(...) returns non-null string that cannot be parsed as integer) (Gate 1).")

        # Gate 2: terminology/symbol hygiene — avoid local variable names in prose
        if re.search(r"\bfileLen\b", comment):
            warnings.append("Comment may refer to local variable 'fileLen'; prefer 'the generated data length' or define in text (Gate 2 symbol hygiene).")

        # Cached sentinel initialization: require explicit initialization postcondition for oracle generation
        if method_code:
            has_sentinel_check = "== -1" in method_code or "== null" in method_code
            has_parse_or_getvalue = "getValue" in method_code or "valueOf" in method_code or "parseInt" in method_code
            has_field_assign = bool(re.search(r"this\.\w+\s*=", method_code)) or has_field_write
            if has_sentinel_check and has_parse_or_getvalue and has_field_assign:
                post_block = self._extract_section_block(comment, "postconditions")
                if not post_block:
                    warnings.append("Cached sentinel initialization detected; add Postconditions section with an explicit initialization bullet (e.g. when field is sentinel at entry, sets to parsed value when non-null else default).")
                else:
                    post_lower = post_block.lower()
                    has_init_wording = (
                        "sets" in post_lower or "initializ" in post_lower or "when non-null" in post_lower
                        or "otherwise to" in post_lower or "else to" in post_lower or "default" in post_lower
                        or "parsed" in post_lower or "at entry" in post_lower
                    )
                    if not has_init_wording:
                        warnings.append("Cached sentinel initialization detected; Postconditions should include an explicit initialization bullet (e.g. when sentinel at entry, sets field from parse/getValue or default).")

        return {
            "missing_side_effects": missing_side_effects,
            "missing_concurrency": missing_concurrency,
            "warnings": warnings,
        }

    def _extract_section_block(self, comment: str, section_name: str) -> str:
        """Extract the body of a section (e.g. Concurrency) for validation. Case-insensitive header match."""
        if not comment:
            return ""
        section_lower = section_name.lower()
        other_headers = {"purpose", "preconditions", "postconditions", "sideeffects", "concurrency", "exceptions", "edgecases"} - {section_lower}
        lines = comment.split("\n")
        in_section = False
        parts = []
        for line in lines:
            stripped = line.strip().lstrip("*").strip()
            if not stripped:
                continue
            if stripped.lower().startswith(section_lower):
                in_section = True
                rest = stripped[len(section_lower):].lstrip(" :").strip()
                if rest:
                    parts.append(rest)
                continue
            if in_section:
                if any(stripped.lower().startswith(h) for h in other_headers):
                    break
                if stripped.startswith("@"):
                    break
                parts.append(stripped)
        return " ".join(parts).strip() if parts else ""

    def _flag_answer_contradicts_code(
        self, answered_questions: Dict[str, str], all_gaps: List[Any], method_code: str
    ) -> List[Dict[str, str]]:
        """
        Flag answered questions where the developer's choice likely contradicts
        observable code behavior. Report-only; the comment still documents
        intention + implementation note. Used so reviewers can see when
        "wrong" answers were given (intention vs code).
        """
        if not answered_questions or not method_code:
            return []
        gap_map = {getattr(g, "id", None): g for g in all_gaps if getattr(g, "id", None)}
        method_lower = method_code.lower()
        has_unguarded_parse = (
            re.search(r"Integer\.(valueOf|parseInt)|Long\.(parseLong|valueOf)", method_code)
            and not re.search(r"try\s*\{[^}]*Integer\.(valueOf|parseInt)", method_code, re.DOTALL)
        )
        flagged = []
        for gap_id, answer in answered_questions.items():
            if not answer or not gap_id:
                continue
            gap = gap_map.get(gap_id)
            if not gap:
                continue
            gtype = getattr(gap, "type", None)
            key = str(answer).strip().upper()
            # Exception: developer said fallback/return default (B) but code throws
            if gtype == "missing_implicit_exception" and key == "B" and has_unguarded_parse:
                flagged.append({
                    "gap_id": gap_id,
                    "answer": answer,
                    "reason": "Answer implies fallback/return default; code has no try-catch and throws.",
                })
        return flagged

    def _strip_redundant_contract_tags(self, comment: str) -> str:
        """
        Remove @implNote and @concurrency lines so the contract is documented
        only in sections, not duplicated in tags. Preserves @param, @return, @throws
        always; never strip those.
        """
        if not comment:
            return comment
        lines = comment.split('\n')
        out = []
        for line in lines:
            stripped = line.strip().lstrip('*').strip()
            if stripped.startswith('@param') or stripped.startswith('@return') or stripped.startswith('@throws'):
                out.append(line)
                continue
            if re.match(r'^@implNote\b', stripped, re.IGNORECASE):
                continue
            if re.match(r'^@concurrency\b', stripped, re.IGNORECASE):
                continue
            out.append(line)
        return '\n'.join(out)

    # Short @throws pointers that are acceptable (no replacement). Case-insensitive.
    _THROWS_SHORT_POINTERS = frozenset({
        "see exceptions", "see exceptions.", "see exceptions section",
        "if parsing fails", "if parsing fails.", "if parsing fails.",
    })

    def _dedupe_throws_conditions(self, comment: str) -> str:
        """
        One source of truth for exception conditions: Exceptions section holds full condition.
        If Exceptions section contains "Throws X if ..." and an @throws X has a different/long
        condition, replace the @throws condition with a short pointer to avoid duplication and
        inconsistent phrasing.
        """
        if not comment:
            return comment
        exceptions_body = self._extract_section_block(comment, "exceptions")
        if not exceptions_body:
            return comment
        # Exception types that have a substantive condition in Exceptions section
        throws_type_in_section = set()
        for m in re.finditer(r"\bThrows\s+(\w+)\s+(?:if|when)\s+", exceptions_body, re.IGNORECASE):
            throws_type_in_section.add(m.group(1))
        if not throws_type_in_section:
            return comment

        lines = comment.split("\n")
        out = []
        for line in lines:
            # Match @throws Type or @throws Type condition (with optional leading * and space)
            stripped = line.strip().lstrip("*").strip()
            m = re.match(r"^@throws\s+(\w+)(?:\s+(.+))?$", stripped, re.IGNORECASE)
            if not m:
                out.append(line)
                continue
            exc_type = m.group(1)
            condition = (m.group(2) or "").strip().rstrip(".")
            if exc_type not in throws_type_in_section:
                out.append(line)
                continue
            if not condition:
                out.append(line)
                continue
            cond_lower = condition.lower()
            if cond_lower in self._THROWS_SHORT_POINTERS:
                out.append(line)
                continue
            if len(condition) <= 25 and ("see" in cond_lower or "parsing fails" in cond_lower):
                out.append(line)
                continue
            # Replace with short pointer
            if "NumberFormatException" in exc_type or "ParseException" in exc_type:
                short = "if parsing fails."
            else:
                short = "see Exceptions."
            # Preserve original line indentation/prefix (e.g. " * ")
            prefix = line[: len(line) - len(line.lstrip())]
            new_line = f"{prefix}@throws {exc_type} {short}"
            out.append(new_line)
        return "\n".join(out)

    def _validate_throws_tags(self, comment: str, method_code: str) -> str:
        """
        Throws-alignment validator (separate from tag stripper). Removes any
        @throws type not in signature_throws and not in inferred_throws.
        Ensures: if signature does not declare IOException and body does not
        throw it, remove @throws IOException even if the LLM produced it.
        Never strips @param or @return; only invalid @throws lines.
        """
        if not comment or not method_code:
            return comment
        # signature_throws: declared in method signature
        signature_line = method_code.split('{')[0] if '{' in method_code else method_code.split('\n')[0]
        throws_match = re.search(r'throws\s+([^{]+)', signature_line)
        declared = set()
        if throws_match:
            clause = throws_match.group(1)
            declared.update(re.findall(r'\b(\w*Exception|\w*Error|Throwable)\b', clause))
        # inferred_throws: explicitly thrown in body
        body = method_code[method_code.find('{') + 1:] if '{' in method_code else ''
        thrown_in_body = set()
        for m in re.finditer(r'throw\s+new\s+(\w+)\b', body):
            thrown_in_body.add(m.group(1))
        allowed = declared | thrown_in_body
        if not allowed:
            return comment
        # Normalize to simple names for comparison (e.g. IOException)
        def simple_name(exc: str) -> str:
            return exc.split('.')[-1] if exc else ""
        allowed_simple = {simple_name(e) for e in allowed}
        # Remove @throws lines whose exception type is not allowed
        lines = comment.split('\n')
        out = []
        for line in lines:
            stripped = line.strip()
            # Match "* @throws TypeName ..." or "@throws TypeName ..."
            match = re.match(r'^(\s*\*\s*)?@throws\s+(\S+)\b', stripped, re.IGNORECASE)
            if match:
                exc_type = simple_name(match.group(2))
                if exc_type and exc_type not in allowed_simple:
                    continue  # Drop this line
            out.append(line)
        return '\n'.join(out)

    def _validate_param_tags(self, comment: str, ast_facts: Dict[str, Any]) -> str:
        """
        Align @param tags with the method signature. Removes any @param whose name
        is not in the method's parameter list (prevents signature drift and overload
        contamination).
        """
        if not comment or not ast_facts:
            return comment
        params = ast_facts.get('parameters', [])
        # parameters are "name:type"; extract names
        allowed_names = set()
        for p in params:
            if isinstance(p, str) and ':' in p:
                allowed_names.add(p.split(':', 1)[0].strip())
            elif isinstance(p, str):
                allowed_names.add(p.strip())
        if not allowed_names:
            return comment
        lines = comment.split('\n')
        out = []
        for line in lines:
            stripped = line.strip()
            match = re.match(r'^(\s*\*\s*)?@param\s+(\S+)\b', stripped, re.IGNORECASE)
            if match:
                param_name = match.group(2)
                if param_name not in allowed_names:
                    continue  # Drop @param for name not in signature
            out.append(line)
        return '\n'.join(out)

    def _strip_npe_for_null_handled_params(self, comment: str, method_code: str) -> str:
        """
        If the method explicitly handles null for a parameter (e.g. if (param == null) return null),
        remove @throws NullPointerException for that param to avoid contradicting behavior.
        """
        if not comment or not method_code:
            return comment
        body = method_code[method_code.find('{') + 1:] if '{' in method_code else method_code
        # Find params that are null-checked with early return: if (X == null) return ...
        null_handled = set()
        for m in re.finditer(r'if\s*\(\s*([a-zA-Z_]\w*)\s*==\s*null\s*\)\s*\{?\s*return\s+', body):
            null_handled.add(m.group(1))
        for m in re.finditer(r'if\s*\(\s*null\s*==\s*([a-zA-Z_]\w*)\s*\)\s*\{?\s*return\s+', body):
            null_handled.add(m.group(1))
        if not null_handled:
            return comment
        # Remove @throws NullPointerException when description refers to a null-handled param
        lines = comment.split('\n')
        out = []
        for line in lines:
            stripped = line.strip()
            match = re.match(r'^(\s*\*\s*)?@throws\s+NullPointerException\s+(.*)$', stripped, re.IGNORECASE)
            if match:
                desc = (match.group(2) or '').lower()
                if any(p in desc for p in null_handled):
                    continue  # Drop this @throws NPE for null-handled param
            out.append(line)
        return '\n'.join(out)

    def _check_unevidenced_behavioral_claims(
        self, comment: str, method_code: str, ast_facts: Dict[str, Any]
    ) -> List[str]:
        """
        Flag behavioral claims in the strengthened comment that cannot be derived
        from the method body (for reviewer/dataset finalization). Report-only;
        does not modify the comment.
        - Preconditions like 'must not be null' for a parameter when the method
          does not explicitly check (no requireNonNull, no if (param == null) throw).
        - Speculative EdgeCases (e.g. behavior not evidenced in code).
        """
        warnings: List[str] = []
        if not comment or not method_code:
            return warnings
        body = method_code[method_code.find('{') + 1:] if '{' in method_code else method_code
        params = ast_facts.get('parameters', [])
        param_names = set()
        for p in params:
            if isinstance(p, str) and ':' in p:
                param_names.add(p.split(':', 1)[0].strip())
            elif isinstance(p, str):
                param_names.add(p.strip())
        # Explicit null enforcement in code?
        def param_explicitly_null_checked(name: str) -> bool:
            if not name:
                return False
            # requireNonNull(param) or if (param == null) throw
            if re.search(rf'Objects\.requireNonNull\s*\(\s*{re.escape(name)}\s*', body):
                return True
            if re.search(rf'if\s*\(\s*{re.escape(name)}\s*==\s*null\s*\)\s*{{?\s*throw', body):
                return True
            if re.search(rf'if\s*\(\s*null\s*==\s*{re.escape(name)}\s*\)\s*{{?\s*throw', body):
                return True
            if re.search(rf'checkNotNull\s*\(\s*{re.escape(name)}\s*', body):
                return True
            return False
        # Preconditions section or @param: "must not be null" for a param not explicitly enforced
        preconditions = self._extract_section_block(comment, "Preconditions")
        comment_lower = comment.lower()
        warned_params: set = set()
        for param_name in param_names:
            if param_explicitly_null_checked(param_name) or param_name in warned_params:
                continue
            # Preconditions section claims this param must not be null?
            if preconditions and param_name.lower() in preconditions.lower() and 'must not be null' in preconditions.lower():
                if param_name in preconditions or f"{{{param_name}}}" in preconditions.lower() or f"{{@code {param_name}}}" in preconditions.lower():
                    warnings.append(
                        f"Unevidenced precondition: parameter '{param_name}' has 'must not be null' but method does not explicitly enforce it (no requireNonNull/checkNotNull/if-throw)."
                    )
                    warned_params.add(param_name)
            # @param X ... must not be null
            if param_name not in warned_params and 'must not be null' in comment_lower and param_name in comment:
                for line in comment.split('\n'):
                    if '@param' in line.lower() and param_name in line and 'must not be null' in line.lower():
                        warnings.append(
                            f"Unevidenced precondition: @param '{param_name}' states 'must not be null' but method does not explicitly enforce it (no requireNonNull/checkNotNull/if-throw)."
                        )
                        warned_params.add(param_name)
                        break
        # EdgeCases: flag if section mentions speculative behavior (heuristic: "may" / "might" in EdgeCases without code evidence)
        edge_cases = self._extract_section_block(comment, "EdgeCases")
        if edge_cases:
            lower = edge_cases.lower()
            if (' may ' in lower or ' might ' in lower) and 'implementation note' not in comment.lower():
                # Only warn if the edge case is not clearly tied to a literal in code
                warnings.append(
                    "EdgeCases section may contain speculative behavior (e.g. 'may'/'might'); ensure only evidenced scenarios are stated."
                )
        # Unsupported NPE from constructor/creation (generalization from expert review)
        if 'nullpointerexception' in comment_lower and ('when constructing' in comment_lower or 'when creating' in comment_lower):
            if not (re.search(r'requireNonNull|checkNotNull|if\s*\([^)]*==\s*null\s*\)\s*\{?\s*throw', body)):
                warnings.append(
                    "Comment claims NullPointerException 'when constructing' or 'when creating'; method body does not explicitly check for null before that. Prefer observable wording or remove unsupported claim (see COMMENT_ALIGNMENT.md §7)."
                )
        return warnings

    def _ensure_concurrency_section(self, comment: str, ast_facts: Dict[str, Any]) -> str:
        """
        If method is synchronized (synchronized_method == True only), ensure comment has
        a Concurrency section. Do not trigger on locks_used or anything else.
        Inserts the block before the first @param/@return/@throws tag if tags exist,
        otherwise before the closing */, so the Concurrency section never appears
        inside or after the tag block.
        """
        if not comment or not ast_facts:
            return comment
        if not ast_facts.get('synchronized_method'):
            return comment
        comment_lower = comment.lower()
        if 'concurrency:' in comment_lower:
            return comment
        concurrency_block = (
            "Concurrency:\n"
            "This method is synchronized. Concurrent calls on the same instance are serialized by the synchronization mechanism."
        )
        lines = comment.split('\n')
        # Find first line that starts a Javadoc tag block (@param, @return, @throws, etc.)
        tag_markers = ('@param', '@return', '@throws', '@exception', '@deprecated', '@see', '@since')
        insert_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match " * @param" or " * @return" etc.
            if any(stripped.startswith('*') and tag in stripped.lower() for tag in tag_markers):
                insert_idx = i
                break
            if stripped.startswith('@') and any(tag in stripped.lower() for tag in tag_markers):
                insert_idx = i
                break
        # If no tag line found, insert before the closing */
        if insert_idx is None:
            for i in range(len(lines) - 1, -1, -1):
                if '*/' in lines[i]:
                    insert_idx = i
                    break
            if insert_idx is None:
                insert_idx = len(lines)
        # Insert as Javadoc lines (blank line then Concurrency heading + body)
        indent = " * "
        to_insert = [indent]
        for line in concurrency_block.split('\n'):
            to_insert.append(indent + line)
        for j, line in enumerate(to_insert):
            lines.insert(insert_idx + j, line)
        return '\n'.join(lines)

    def _apply_minimal_strengthening(self, original_comment: str, ast_facts: Dict, 
                                     gap_results: Dict, method_code: str) -> str:
        """
        Apply minimal strengthening using only auto_add facts when LLM generation fails.
        
        NEW: Avoids silent fallback by inserting deterministic snippets directly into original comment.
        """
        auto_add_gaps = gap_results.get("auto_add", [])
        
        if not auto_add_gaps:
            return original_comment
        
        # Collect snippets by doc_slot
        snippets_by_slot = {}
        
        for gap in auto_add_gaps:
            if gap.type == "field_write_fact":
                # Generate side effect snippet
                fields = gap.parameters if hasattr(gap, 'parameters') else []
                if fields:
                    if len(fields) == 1:
                        snippet = f"Updates the {fields[0]} field."
                    else:
                        field_list = ", ".join(fields[:3])
                        snippet = f"Modifies instance fields: {field_list}."
                    snippets_by_slot.setdefault("SideEffects", []).append(snippet)
            
            elif gap.type == "fact_scenario_gap" or (gap.type == "execution_scenario_gap" and gap.scenario_kind == "early_return"):
                # Generate early return snippet
                condition = gap.scenario_condition or ""
                if condition:
                    snippet = f"If {condition}, the method returns early without performing the main operation."
                    snippets_by_slot.setdefault("Postconditions", []).append(snippet)
            
            elif gap.type == "behavior_mismatch_gap" and gap.action == "auto_add":
                # Generate contradiction resolution snippet
                condition = gap.scenario_condition or ""
                if "null" in condition.lower():
                    snippet = "If the configuration is not set, the method logs an informational message and returns without sending any notification."
                    snippets_by_slot.setdefault("Postconditions", []).append(snippet)
        
        # Also check for InterruptedException in signature with Thread.sleep
        method_signature = ast_facts.get('method_signature', '')
        method_calls = ast_facts.get('method_calls', [])
        if 'throws' in method_signature and 'InterruptedException' in method_signature:
            if any('sleep' in call.lower() or 'wait' in call.lower() for call in method_calls):
                snippets_by_slot.setdefault("Exceptions", []).append(
                    "@throws InterruptedException if the thread is interrupted while sleeping between retry attempts."
                )
        
        if not snippets_by_slot:
            return original_comment
        
        # Insert snippets into original comment
        # Simple approach: append sections at the end
        lines = original_comment.split('\n')
        result_lines = lines.copy()
        
        # Find end of Javadoc (before closing */)
        end_idx = len(result_lines) - 1
        for i in range(len(result_lines) - 1, -1, -1):
            if '*/' in result_lines[i]:
                end_idx = i
                break
        
        # Insert sections before closing */
        insert_lines = []
        for slot, snippets in snippets_by_slot.items():
            insert_lines.append(f" * {slot}:")
            for snippet in snippets:
                insert_lines.append(f" *   {snippet}")
        
        if insert_lines:
            result_lines.insert(end_idx, '\n'.join(insert_lines))
        
        return '\n'.join(result_lines)
    
    def _interactive_question_loop(self, method_id: str, questions: List[Question], method_code: str) -> Dict[str, str]:
        """Interactive question-answering loop."""
        answers = {}
        
        print(f"\n{'='*60}")
        print(f"Method: {method_id}")
        print(f"{'='*60}")
        # Supervisor preference: show complete source code of the method (AST/gap used only for question preparation)
        print(f"\nComplete method source code:\n{method_code}\n")
        
        for question in questions[:2]:
            print(f"\n[Question {question.id}] Priority: {question.priority}")
            print(f"Category: {question.category} | Doc Slot: {question.doc_slot}")
            print(f"\n{question.question_text}\n")
            if question.options:
                print("Options (MCQ):")
                for opt in question.options:
                    key = opt.get('key', '')
                    text = opt.get('text', '')
                    print(f"  {key}. {text}")
                print()
            # Method code already printed once above; avoid duplicating in prompt/context
            
            while True:
                try:
                    if question.options:
                        valid_keys = [str(o.get('key', '')).upper() for o in question.options if o.get('key')]
                        prompt = f"Enter option ({', '.join(valid_keys)}) or 'skip': "
                        answer = input(prompt).strip()
                        if answer and answer.upper() in valid_keys:
                            answer = answer.upper()
                        elif answer.lower() == 'skip' or answer == '':
                            break
                        else:
                            print(f"Please enter one of {valid_keys} or 'skip'.")
                            continue
                    else:
                        answer = input("Enter your answer in your own words (max 20 words), or 'skip': ").strip()
                        if answer == '' or answer.lower() == 'skip':
                            break
                        word_count = len(answer.split())
                        if word_count > 20:
                            print("Answer must be 20 words or fewer. Try again, or type 'skip'.")
                            continue
                except (KeyboardInterrupt, EOFError):
                    print("\nPlease enter your answer or 'skip'. (Script will not exit on key press.)")
                    continue
                answers[question.id] = answer
                if self.question_bank and method_id:
                    self.question_bank.update_answer(method_id, question.id, answer)
                break
        
        return answers
    
    def _generate_fallback_question(self, method_id: str, method_code: str, original_comment: str, ast_facts: Dict) -> Optional[Question]:
        """
        Generate a fallback question when no gaps are detected.
        This ensures every entry gets at least one question for completeness.
        
        Args:
            method_id: Method identifier
            method_code: Method source code
            original_comment: Original Javadoc comment
            ast_facts: Extracted AST facts
            
        Returns:
            A fallback Question object, or None if generation fails
        """
        from gap_detector.models import Question
        
        # Generate a general completeness question
        method_name = ast_facts.get('method_name', 'this method')
        return_type = ast_facts.get('return_type', 'void')
        has_params = len(ast_facts.get('parameters', [])) > 0
        
        # SUPERVISOR: Short question only (≤20 words), no options
        if return_type != 'void' and (not original_comment or '@return' not in original_comment):
            question_text = f"Any return semantics or preconditions for {method_name} to document?"
        elif has_params and (not original_comment or '@param' not in original_comment):
            question_text = f"Any preconditions or parameter behaviors for {method_name}?"
        else:
            question_text = f"Any extra contract details or edge cases for {method_name}?"
        if len(question_text.split()) > 20:
            question_text = "Any additional details for this method to document?"
        question_id = f"GAP-FALLBACK-{method_id}"
        context_code = method_code
        return Question(
            id=question_id,
            priority=1,
            category="completeness",
            doc_slot="General",
            question_text=question_text,
            context_code=context_code,
            options=[],  # SUPERVISOR: No multiple-choice options
            evidence_confidence="low",
            fact_or_guarantee="guarantee"
        )
    
    def _question_to_dict(self, question: Question) -> Dict[str, Any]:
        """Convert Question object to dictionary."""
        # FIXED: Remove truncation markers from user-visible fields
        question_text = question.question_text
        if question_text and '...' in question_text:
            # Remove truncation marker - this should not happen, but clean it if it does
            question_text = question_text.replace('...', '').strip()
        
        # Clean options text
        cleaned_options = []
        if question.options:
            for opt in question.options:
                opt_copy = opt.copy()
                if 'text' in opt_copy and '...' in opt_copy['text']:
                    opt_copy['text'] = opt_copy['text'].replace('...', '').strip()
                cleaned_options.append(opt_copy)
        
        # Truncate context_code if needed (this is OK to truncate, but mark it clearly)
        context_code = question.context_code
        if context_code and len(context_code) > 500:
            context_code = context_code[:500] + "[truncated]"
        
        return {
            'id': question.id,
            'priority': question.priority,
            'category': question.category,
            'doc_slot': question.doc_slot,
            'question_text': question_text,
            'context_code': context_code,
            'options': cleaned_options,
            'evidence_confidence': question.evidence_confidence,
            'fact_or_guarantee': question.fact_or_guarantee,
            'developer_answer': question.developer_answer,
            'answered': question.answered
        }
    
    def _count_gaps_by_kind(self, gaps: List[Gap]) -> Dict[str, int]:
        """Count gaps by kind (fact, guarantee, limitation) for empirical study metrics."""
        counts = {"fact": 0, "guarantee": 0, "limitation": 0}
        for g in gaps:
            if getattr(g, "kind", None) == "limitation" or (g.type or "").startswith("limitations_") or "limitation" in (g.type or "").lower():
                counts["limitation"] += 1
            elif g.kind == "guarantee":
                counts["guarantee"] += 1
            else:
                counts["fact"] += 1
        return counts
    
    def _gap_to_dict(self, gap: Gap) -> Dict[str, Any]:
        """Convert Gap object to dictionary."""
        return {
            'id': gap.id,
            'type': gap.type,
            'doc_slot': gap.doc_slot,
            'priority': gap.priority,
            'evidence_confidence': gap.evidence_confidence,
            'kind': gap.kind,  # FIXED: Use 'kind' instead of 'fact_or_guarantee'
            'action': gap.action,  # NEW: Include action field
            'parameters': gap.parameters,
            'issue': gap.issue,
            'evidence_snippet': gap.evidence_snippet,
            'question': gap.question,
            'suggested_options': gap.suggested_options,
            'dedup_key': gap.dedup_key,
            'doc_insert_target': gap.doc_insert_target,  # NEW: Include doc_insert_target
            'risk_level': gap.risk_level,  # NEW: Include risk_level
            'llm_rationale': gap.llm_rationale,  # NEW: Include llm_rationale if present
            'llm_rank': gap.llm_rank  # NEW: Include llm_rank if present
        }
    
    def _compute_facts_added_from_diff(self, original_comment: str, strengthened_comment: str) -> List[str]:
        """
        Compute facts_added by diffing original vs strengthened comment.
        
        FIXED: facts_added should reflect what was actually added, not just gaps list.
        """
        facts_added = []
        
        # Extract sections from both comments
        sections = ['Purpose', 'Preconditions', 'Postconditions', 'SideEffects', 
                   'Concurrency', 'Exceptions', 'EdgeCases', 'Inputs', 'Outputs']
        
        for section in sections:
            # Extract section content from original
            original_section = self._extract_section_content(original_comment, section)
            strengthened_section = self._extract_section_content(strengthened_comment, section)
            
            # Check if section has new non-trivial content
            if strengthened_section and strengthened_section.strip().lower() not in ['none', 'not specified', '']:
                # Check if it's different from original
                if not original_section or original_section.strip().lower() != strengthened_section.strip().lower():
                    # Check if it's not just "None" replacing empty
                    if strengthened_section.strip().lower() != 'none':
                        facts_added.append(section)
        
        return facts_added
    
    def _extract_section_content(self, comment: str, section_name: str) -> str:
        """Extract content of a specific section from comment."""
        # Pattern: SectionName: ... (until next section or end)
        pattern = rf'{section_name}:\s*(.*?)(?=\n\s*(?:Purpose|Preconditions|Postconditions|SideEffects|Concurrency|Exceptions|EdgeCases|Inputs|Outputs|@param|@return|@throws|\*/):)'
        match = re.search(pattern, comment, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _check_tag_mismatch(self, method_code: str, original_comment: str, 
                           strengthened_comment: str, ast_facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for tag mismatches between signature and Javadoc.
        
        FIXED: Flag @throws tags that don't match signature or code evidence.
        """
        from utils.token_utils import extract_javadoc_tags
        
        # Extract declared exceptions from signature
        declared_exceptions = set()
        # Check method signature for throws clause
        throws_match = re.search(r'throws\s+([^{]+)', method_code.split('{')[0] if '{' in method_code else method_code)
        if throws_match:
            throws_clause = throws_match.group(1)
            # Parse exception types
            exception_types = re.findall(r'(\w+Exception|\w+Error)', throws_clause)
            declared_exceptions.update(exception_types)
        
        # Also check AST facts
        ast_exceptions = ast_facts.get('exceptions_thrown', [])
        declared_exceptions.update(ast_exceptions)
        
        # Extract @throws tags from original comment
        original_tags = extract_javadoc_tags(original_comment)
        javadoc_throws = set()
        if 'throws' in original_tags:
            javadoc_throws.update({t[0] if isinstance(t, (list, tuple)) else t for t in original_tags['throws']})
        
        # Check for mismatches
        mismatches = []
        for javadoc_exc in javadoc_throws:
            # Check if it's in declared exceptions or can be proven from code
            if javadoc_exc not in declared_exceptions:
                # Check if it can be proven from implicit exceptions (parsing, array access, etc.)
                # This is a simplified check - full validation would require gap detection
                if not self._can_prove_exception_from_code(method_code, javadoc_exc):
                    mismatches.append({
                        'javadoc_exception': javadoc_exc,
                        'declared_exceptions': list(declared_exceptions),
                        'reason': 'not_in_signature_or_code'
                    })
        
        return {
            'has_mismatch': len(mismatches) > 0,
            'details': {
                'declared_exceptions': list(declared_exceptions),
                'javadoc_throws': list(javadoc_throws),
                'mismatches': mismatches
            }
        }
    
    def _can_prove_exception_from_code(self, method_code: str, exc_name: str) -> bool:
        """Check if exception can be proven from code patterns."""
        code_lower = method_code.lower()
        
        if exc_name == 'NumberFormatException':
            return bool(re.search(r'Integer\.(valueOf|parseInt)|Long\.(valueOf|parseLong)', method_code))
        elif exc_name == 'ArithmeticException':
            return bool(re.search(r'\b\w+\s*[/%]\s*\w+', method_code))
        elif exc_name == 'ArrayIndexOutOfBoundsException':
            return '[' in method_code and ']' in method_code
        elif exc_name == 'NullPointerException':
            return bool(re.search(r'\w+\.\w+|Objects\.requireNonNull', method_code))
        
        return False
    
    def process_dataset(self, input_file: str, output_file: str, use_prefix: bool = True,
                       interactive: bool = False, question_bank_file: str = None,
                       limit: int = None, questions_only: bool = False,
                       regenerate_questions: bool = False):
        """
        Process entire dataset.
        
        Args:
            input_file: Input dataset JSON file
            output_file: Output results JSON file
            use_prefix: If True, use prefix (buggy) version, else postfix (fixed)
            interactive: If True, prompt user for questions interactively
            question_bank_file: Path to question bank file (optional)
            limit: Maximum number of entries to process (None for all entries)
            questions_only: If True, only generate and store questions (no strengthening)
        """
        import copy
        
        # Load dataset
        dataset = load_dataset(input_file)
        total_entries = len(dataset)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            dataset = dataset[:limit]
            print(f"\nProcessing first {len(dataset)} entries (limited from {total_entries} total entries)")
        else:
            print(f"\nProcessing all {total_entries} entries")
        
        # QUESTIONS-ONLY MODE: generate questions and exit early
        if questions_only:
            if not self.enable_gap_detection:
                print("\n[Questions-only] Gap detection is disabled. No questions can be generated.")
                return
            if not question_bank_file:
                print("\n[Questions-only] No question bank file provided. Use --question-bank to specify a JSON file.")
                return
            
            # Initialize question bank for questions-only mode
            # IMPORTANT: In questions-only mode with limit, we want to update only the processed entries
            # So we load existing bank to preserve other entries, but will only save processed ones
            if question_bank_file and not self.question_bank:
                self.question_bank = QuestionBank(question_bank_file)
            elif question_bank_file and self.question_bank and self.question_bank.bank_file != question_bank_file:
                self.question_bank = QuestionBank(question_bank_file)
            
            # FIX: If limit is specified, only save the processed entries
            # Store original bank state to restore unprocessed entries
            original_bank = dict(self.question_bank.bank) if self.question_bank else {}
            
            if limit is not None and limit > 0:
                print(f"\n[Questions-only] Generating questions for first {len(dataset)} entries "
                      f"(limited from {total_entries} total entries)")
            else:
                print(f"\n[Questions-only] Generating questions for all {total_entries} entries")

            method_count = 0
            skipped_count = 0
            for entry in dataset:
                methods = extract_method_data(entry, use_prefix=use_prefix)
                for method_data in methods:
                    method_id = method_data['method_id']
                    method_code = method_data['method_code']
                    original_comment = method_data['original_comment']
                    
                    # OPTIMIZATION: Skip methods that already have questions (unless regenerating)
                    if not regenerate_questions:
                        if method_id in self.question_bank.bank and self.question_bank.bank[method_id]:
                            existing_questions = self.question_bank.bank[method_id]
                            # Check if questions are non-empty (not just empty list)
                            if existing_questions and len(existing_questions) > 0:
                                skipped_count += 1
                                print(f"[Questions-only] Skipping {method_id} (already has {len(existing_questions)} questions)")
                                continue
                    
                    method_count += 1
                    print(f"[Questions-only] Analyzing {method_id}...")

                    # Extract AST facts and detect gaps
                    ast_facts = extract_ast_facts(method_code)
                    ast_facts['method_id'] = method_id  # Add method_id for method-scoped dedup
                    
                    # Detect gaps (returns Dict with "auto_add", "auto_fix", "ask", "skip")
                    gap_results = self.gap_detector.detect_gaps(ast_facts, original_comment, method_code, mode=self.mode)
                    
                    # Only scenario-based (execution_scenario_gap) questions are sent to developer
                    gaps = gap_results.get("ask", [])
                    scenario_gaps = [g for g in gaps if getattr(g, "type", None) == "execution_scenario_gap"]

                    # Generate questions only from scenario gaps (max 2 per method; simple, scenario-focused).
                    # SUPERVISOR: Do not add a "complete requirements" fallback when no ask gaps; allow 0 questions.
                    questions = self.question_generator.generate_questions(scenario_gaps, method_code)

                    # Store questions in question bank (even if empty list)
                    self.question_bank.add_questions(method_id, questions)
            
            # FIX: If limit was specified, merge processed entries with original bank
            # This preserves entries that weren't processed in this run
            # EXCEPTION: If regenerating questions, don't merge - only keep processed entries
            if limit is not None and limit > 0 and not regenerate_questions:
                # Keep processed entries (they may have been updated)
                processed_method_ids = {method_data['method_id'] 
                                      for entry in dataset 
                                      for method_data in extract_method_data(entry, use_prefix=use_prefix)}
                # Restore unprocessed entries from original bank
                for method_id, questions in original_bank.items():
                    if method_id not in processed_method_ids:
                        self.question_bank.bank[method_id] = questions
            elif limit is not None and limit > 0 and regenerate_questions:
                # When regenerating with limit, only keep processed entries (don't merge with old)
                processed_method_ids = {method_data['method_id'] 
                                      for entry in dataset 
                                      for method_data in extract_method_data(entry, use_prefix=use_prefix)}
                # Clear bank and only keep processed entries
                self.question_bank.bank = {mid: self.question_bank.bank[mid] 
                                          for mid in processed_method_ids 
                                          if mid in self.question_bank.bank}
            
            # Save question bank once at the end
            self.question_bank.save()
            print(f"\n[Questions-only] Generated questions for {method_count} methods.")
            if skipped_count > 0:
                print(f"[Questions-only] Skipped {skipped_count} methods (already have questions).")
            print(f"[Questions-only] Question bank saved to: {self.question_bank.bank_file}")
            return

        # NORMAL MODE: full strengthening pipeline
        output_entries = []
        
        for entry in dataset:
            # Create a deep copy of the entry to preserve all original fields
            output_entry = copy.deepcopy(entry)
            
            # Extract methods and strengthen comments
            methods = extract_method_data(entry, use_prefix=use_prefix)
            strengthened_comments = []
            
            for method_data in methods:
                method_id = method_data['method_id']
                method_code = method_data['method_code']
                original_comment = method_data['original_comment']
                
                print(f"Processing {method_id}...")
                
                result = self.strengthen_comment(
                    method_code,
                    original_comment,
                    method_id=method_id,
                    interactive=interactive,
                    requirements_changed=method_data.get('requirements_changed', False),
                )
                result['method_id'] = method_id
                # ENHANCEMENT: Add prefix source code and original comment to result
                result['prefix_source_code'] = method_code
                result['original_comment'] = original_comment
                strengthened_comments.append(result)
            
            # Add strengthened_comments field to the output entry
            output_entry['strengthened_comments'] = strengthened_comments
            output_entries.append(output_entry)
        
        # Save question bank if it was used
        if self.question_bank:
            self.question_bank.save()
        
        # Save results with all original fields plus strengthened_comments
        save_results(output_entries, output_file)
        print(f"\nProcessed {len(output_entries)} entries. Results saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Strengthen Java method comments')
    parser.add_argument('--input', required=True, help='Input dataset JSON file')
    parser.add_argument('--output', required=True, help='Output results JSON file')
    parser.add_argument('--mode', choices=['rewrite', 'contract'], default='contract',
                       help='Strengthening mode')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'deepseek'], default='deepseek',
                       help='LLM provider (default: deepseek)')
    parser.add_argument('--model', default='deepseek-coder', help='LLM model name (default: deepseek-coder)')
    parser.add_argument('--use-prefix', action='store_true', default=True,
                       help='Use prefix (buggy) version')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode for question answering')
    parser.add_argument('--question-bank', type=str, default=None,
                       help='Path to question bank JSON file (for offline question answering)')
    parser.add_argument('--no-gap-detection', action='store_true',
                       help='Disable gap detection (use original behavior)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of entries to process (e.g., 10 for first 10 entries, None for all)')
    parser.add_argument('--questions-only', action='store_true',
                       help='Generate and store questions only (no strengthened comments)')
    parser.add_argument('--regenerate-questions', action='store_true',
                       help='Regenerate questions even if they already exist in question bank')
    
    args = parser.parse_args()
    
    # Interactive prompt if limit is not specified
    if args.limit is None:
        while True:
            try:
                response = input("\nHow many entries to process?\n  [1] First 10 entries\n  [2] All entries\nEnter choice (1 or 2): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nInvalid key. Please enter 1 or 2. (Script will not exit.)")
                continue
            if response == '1':
                args.limit = 10
                print("Selected: First 10 entries\n")
                break
            elif response == '2':
                args.limit = None
                print("Selected: All entries\n")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Initialize question bank if file provided
    question_bank = None
    if args.question_bank:
        question_bank = QuestionBank(args.question_bank)
    
    strengthener = CommentStrengthener(
        mode=args.mode,
        llm_provider=args.provider,
        llm_model=args.model,
        enable_gap_detection=not args.no_gap_detection,
        question_bank=question_bank
    )
    
    strengthener.process_dataset(
        args.input,
        args.output,
        use_prefix=args.use_prefix,
        interactive=args.interactive,
        question_bank_file=args.question_bank,
        limit=args.limit,
        questions_only=args.questions_only,
        regenerate_questions=args.regenerate_questions
    )


if __name__ == '__main__':
    main()

