"""
Question Bank Module

Manages question bank storage and loading for offline question answering.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from gap_detector.models import Question


class QuestionBank:
    """Manages question bank for offline question answering."""
    
    def __init__(self, bank_file: Optional[str] = None):
        """
        Initialize question bank.
        
        Args:
            bank_file: Path to question bank JSON file (optional)
        """
        self.bank_file = bank_file
        self.bank: Dict[str, List[Dict[str, Any]]] = {}
        if bank_file and Path(bank_file).exists():
            self.load()
    
    def load(self):
        """Load question bank from file."""
        if not self.bank_file:
            return
        
        try:
            with open(self.bank_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert list format to dict format if needed
                if isinstance(data, list):
                    self.bank = {item['method_id']: item.get('questions', []) for item in data}
                else:
                    self.bank = data
        except (FileNotFoundError, json.JSONDecodeError):
            self.bank = {}
    
    def save(self):
        """Save question bank to file."""
        if not self.bank_file:
            return
        
        # Convert to list format for compatibility
        data = [
            {
                'method_id': method_id,
                'questions': questions
            }
            for method_id, questions in self.bank.items()
        ]
        
        with open(self.bank_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_questions(self, method_id: str, questions: List[Question]):
        """Add questions for a method."""
        self.bank[method_id] = [self._question_to_dict(q) for q in questions]
    
    def get_answered_questions(self, method_id: str) -> Dict[str, str]:
        """
        Get answered questions for a method.
        
        Returns:
            Dictionary mapping gap IDs to developer answers
        """
        if method_id not in self.bank:
            return {}
        
        answers = {}
        for q_dict in self.bank[method_id]:
            if q_dict.get('answered', False) and q_dict.get('developer_answer'):
                answers[q_dict['id']] = q_dict['developer_answer']
        
        return answers
    
    def update_answer(self, method_id: str, gap_id: str, answer: str):
        """Update answer for a specific question."""
        if method_id not in self.bank:
            return
        
        for q_dict in self.bank[method_id]:
            if q_dict['id'] == gap_id:
                q_dict['developer_answer'] = answer
                q_dict['answered'] = True
                break
        
        self.save()
    
    def _question_to_dict(self, question: Question) -> Dict[str, Any]:
        """Convert Question object to dictionary."""
        return {
            'id': question.id,
            'priority': question.priority,
            'category': question.category,
            'doc_slot': question.doc_slot,
            'question_text': question.question_text,
            'context_code': question.context_code,
            'options': question.options,
            'evidence_confidence': question.evidence_confidence,
            'fact_or_guarantee': question.fact_or_guarantee,
            'developer_answer': question.developer_answer,
            'answered': question.answered
        }

