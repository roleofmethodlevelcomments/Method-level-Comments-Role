"""
LLM Client Module

Handles LLM API calls with retry logic and deterministic settings.
"""

import os
import time
from typing import List, Optional
from openai import OpenAI
import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """LLM client with retry logic."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai", "anthropic", or "deepseek"
            model: Model name to use
        """
        self.provider = provider.lower()
        self.model = model
        
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.api_key = api_key
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.api_key = api_key
        elif self.provider == "deepseek":
            # Use API key from environment or fallback to the one found in the script
            api_key = os.getenv("DEEPSEEK_API_KEY", "sk-e17da7ffe75e432c824c3ac3c98ad3bb")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment")
            self.api_key = api_key
            self.client = None  # DeepSeek uses direct HTTP requests
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, deepseek")
    
    def generate(self, prompt: str, num_candidates: int = 1, max_retries: int = 3) -> List[str]:
        """
        Generate comment candidates with retry logic.
        
        Args:
            prompt: Input prompt
            num_candidates: Number of candidates to generate
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of generated comment strings
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt, num_candidates)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, num_candidates)
                elif self.provider == "deepseek":
                    return self._generate_deepseek(prompt, num_candidates)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def _generate_openai(self, prompt: str, num_candidates: int) -> List[str]:
        """Generate using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Java documentation specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            n=num_candidates
        )
        
        candidates = []
        for choice in response.choices:
            content = choice.message.content.strip()
            candidates.append(content)
        
        return candidates
    
    def _generate_anthropic(self, prompt: str, num_candidates: int) -> List[str]:
        """Generate using Anthropic API."""
        # Anthropic doesn't support n parameter, so we call multiple times
        candidates = []
        for _ in range(num_candidates):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content[0].text.strip()
            candidates.append(content)
        
        return candidates
    
    def _generate_deepseek(self, prompt: str, num_candidates: int) -> List[str]:
        """Generate using DeepSeek API."""
        # DeepSeek API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # DeepSeek doesn't support n parameter, so we call multiple times
        candidates = []
        for candidate_idx in range(num_candidates):
            data = {
                "model": self.model if self.model else "deepseek-coder",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Java documentation specialist."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 1024
            }
            
            # Retry logic for connection errors
            max_retries = 3
            retry_delay = 2  # Start with 2 seconds
            response = None
            candidate_success = False
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    candidate_success = True
                    break  # Success, exit retry loop
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout,
                        requests.exceptions.SSLError) as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"[LLM Client] Connection error for candidate {candidate_idx + 1}/{num_candidates} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        print(f"[LLM Client] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed - log but continue with next candidate
                        print(f"[LLM Client] Failed to generate candidate {candidate_idx + 1}/{num_candidates} after {max_retries} attempts: {str(e)}")
                        print(f"[LLM Client] Continuing with remaining candidates...")
                        break  # Exit retry loop, but continue to next candidate
                except requests.exceptions.HTTPError as e:
                    # HTTP errors (4xx, 5xx) should not be retried
                    print(f"[LLM Client] HTTP error for candidate {candidate_idx + 1}/{num_candidates}: {str(e)}")
                    print(f"[LLM Client] Skipping this candidate...")
                    break  # Exit retry loop, continue to next candidate
                except Exception as e:
                    # Other unexpected errors
                    print(f"[LLM Client] Unexpected error for candidate {candidate_idx + 1}/{num_candidates}: {str(e)}")
                    print(f"[LLM Client] Skipping this candidate...")
                    break  # Exit retry loop, continue to next candidate
            
            # Process response if we got one
            if candidate_success and response is not None:
                try:
                    result = response.json()
                    if "choices" not in result or len(result["choices"]) == 0:
                        print(f"[LLM Client] Warning: Empty choices in DeepSeek response for candidate {candidate_idx + 1}")
                        continue  # Skip this candidate
                    
                    content = result["choices"][0]["message"]["content"].strip()
                    if not content:
                        print(f"[LLM Client] Warning: Empty content in DeepSeek response for candidate {candidate_idx + 1}")
                        continue  # Skip this candidate
                    
                    candidates.append(content)
                except Exception as e:
                    print(f"[LLM Client] Error processing response for candidate {candidate_idx + 1}: {str(e)}")
                    continue  # Skip this candidate
        
        # If we got at least one candidate, return them; otherwise raise an error
        if not candidates:
            raise requests.exceptions.ConnectionError(
                f"Failed to generate any candidates from DeepSeek API after {num_candidates} attempts"
            )
        
        return candidates

