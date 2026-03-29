"""
LLM Generator Module

Orchestrates LLM generation for comment strengthening.
"""

from typing import List
from .client import LLMClient


class CommentGenerator:
    """Generates strengthened comment candidates using LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initialize comment generator.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name
        """
        self.client = LLMClient(provider=provider, model=model)
    
    def generate_candidates(self, prompt: str, mode: str) -> List[str]:
        """
        Generate comment candidates based on mode.
        
        Args:
            prompt: Generated prompt
            mode: "rewrite" or "contract"
            
        Returns:
            List of candidate comment strings
        """
        # Rewrite mode: 1 candidate
        # Contract mode: 3 candidates
        num_candidates = 1 if mode == "rewrite" else 3
        
        candidates = self.client.generate(prompt, num_candidates=num_candidates)
        
        return candidates

