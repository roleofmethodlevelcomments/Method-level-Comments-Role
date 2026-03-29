"""
Routing Logic for Selective Strengthening

Determines how each detected gap should be handled based on priority,
evidence confidence, and fact vs guarantee distinction.
"""

from typing import List, Dict, Any
from gap_detector.models import Gap, Question


class GapRouter:
    """Routes gaps based on priority and evidence confidence."""
    
    def __init__(self):
        pass
    
    def route_gaps(self, gaps: List[Gap], answered_questions: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Route gaps into categories for documentation generation.
        
        Args:
            gaps: List of detected gaps
            answered_questions: Dictionary mapping gap IDs to developer answers
            
        Returns:
            Dictionary with:
            - facts_to_add: Gaps that can be added as facts
            - guarantees_confirmed: Gaps confirmed by developers
            - candidate_inferences: Medium confidence gaps kept as candidates
            - pending_questions: High priority gaps needing answers
        """
        answered_questions = answered_questions or {}
        
        facts_to_add = []
        guarantees_confirmed = []
        candidate_inferences = []
        pending_questions = []
        
        for gap in gaps:
            gap_id = gap.id
            
            # If developer has answered, use the answer
            if gap_id in answered_questions:
                if gap.kind == "guarantee":
                    guarantees_confirmed.append(gap)
                continue
            
            # Limitation gaps: never block acceptance; treat as hints only (candidate_inferences).
            if (
                getattr(gap, "kind", None) == "limitation"
                or (gap.type or "").startswith("limitations_")
                or "limitation" in (gap.type or "").lower()
            ):
                candidate_inferences.append(gap)
                continue
            
            # Routing logic based on priority, confidence, and type
            if gap.evidence_confidence == "high" and gap.kind == "fact":
                # High confidence facts can be auto-added
                facts_to_add.append(gap)
            
            elif gap.kind == "guarantee":
                # Guarantees require confirmation unless code explicitly enforces
                if self._is_code_enforced_guarantee(gap):
                    # Code explicitly enforces this, can document as fact
                    facts_to_add.append(gap)
                elif gap.priority >= 4:
                    # High priority guarantees need questions
                    pending_questions.append(gap_id)
                elif gap.priority == 3 and self._has_risk_features(gap):
                    # Priority 3 with risk features
                    pending_questions.append(gap_id)
            
            elif gap.evidence_confidence == "medium":
                # Medium confidence: keep as candidate inference, ask if high priority
                if gap.priority >= 4:
                    pending_questions.append(gap_id)
                elif gap.priority == 3 and self._has_risk_features(gap):
                    pending_questions.append(gap_id)
                else:
                    candidate_inferences.append(gap)
            
            elif gap.priority <= 2 and gap.evidence_confidence == "low":
                # Low priority, low confidence: skip
                continue
        
        return {
            "facts_to_add": facts_to_add,
            "guarantees_confirmed": guarantees_confirmed,
            "candidate_inferences": candidate_inferences,
            "pending_questions": pending_questions
        }
    
    def _is_code_enforced_guarantee(self, gap: Gap) -> bool:
        """
        Check if a guarantee is explicitly enforced by code.
        
        Example: Explicit null check that throws exception = code-enforced guarantee.
        """
        # For now, high confidence exceptions are considered code-enforced
        if gap.type == "missing_exception" and gap.evidence_confidence == "high":
            return True
        return False
    
    def _has_risk_features(self, gap: Gap) -> bool:
        """
        Check if gap has risk features that warrant asking priority 3 questions.
        
        Risk features:
        - Writes to instance/static fields
        - IO operations, file, network, database
        - Returns mutable collections or internal state references
        - Uses synchronization, locks, atomics, volatile
        - Starts threads, schedules tasks, registers callbacks
        """
        # This is a simplified check - in full implementation, would analyze AST facts
        risk_types = [
            "missing_side_effect",  # Field writes
            "missing_concurrency",  # Synchronization
            "missing_resource_lifecycle"  # Resource management
        ]
        return gap.type in risk_types
    
    def should_ask_question(self, gap: Gap) -> bool:
        """Determine if a question should be asked for this gap."""
        routing = self.route_gaps([gap])
        return gap.id in routing["pending_questions"]

