#!/usr/bin/env python3
"""
LLM Bug Analysis Script for Google Guava Dataset - Pre/Post-Fix Analysis
========================================================================

This script performs requirement evolution analysis with 4 modes:

MODE 1: buggy_method_sourcecode
- Input: Prefix focal method source code WITHOUT any comments
- Tests: Can LLM detect bugs with code alone?

MODE 2: prefix_method_plus_comment  
- Input: Prefix focal method source code + Prefix method-level comments
- Tests: Do ORIGINAL requirements help bug detection?

MODE 3: prefix_method_postfix_comments
- Input: Prefix focal method source code (no comments) + Postfix method-level comments
- Tests: Do UPDATED requirements (after bug fix) help bug detection better?

MODE 4: prefix_method_strengthened_comment
- Input: Prefix focal method source code + Strengthened comment (from Comments Strengthener)
- Tests: Do STRENGTHENED requirements help bug detection better?

Supported Datasets:
- Dataset_for_bug_detection.json - pre/post-fix format (prefix/postfix with source_code, comments). Modes 1–3 only; no strengthened comments.
- Dataset_strengthened_for_bug_detection.json - same as above with 'strengthened_comments' (list); supports all 4 modes including Mode 4.
- dataset_strengthened_bug_detection.json - 228 entries; pre/post-fix + 'strengthened_comments' (list); supports all 4 modes.
- code_llama_strengthened.json (228 entries) - uses 'strengthened_comment' field
- strengthened_comments_full.json (244 entries) - uses 'strengthened_comments' field (list)
- strengthened_comments.json - uses 'strengthened_comments' field (list), same format as strengthened_comments_full
- strengthened_results.json - uses 'strengthened_comments' field (list), same format
- strengthened_dataset_interactive.json - uses 'strengthened_comments' field (list), same format
- strengthened_with_answers.json - uses 'strengthened_comments' field (list)
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
import logging
import requests
from requests.exceptions import Timeout as RequestsTimeout, ConnectionError as RequestsConnectionError, RequestException
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set fixed random seed for reproducibility
random.seed(0)

class GuavaPrePostFixLLMAnalyzer:
    """Analyzes buggy methods from pre/post-fix dataset using LLM"""
    
    def __init__(self, input_file: str = "code_llama_strengthened.json", 
                 output_file: str = "llm_analyzed_prepostfix.json",
                 analysis_mode: str = "buggy_method_sourcecode",
                 api_timeout: int = 90,
                 api_max_retries: int = 3):
        """
        Initialize analyzer.
        
        Supports two input formats:
        - code_llama_strengthened.json: Uses 'strengthened_comment' (string) field
        - strengthened_comments_full.json: Uses 'strengthened_comments' (list) field
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            analysis_mode: Analysis mode (buggy_method_sourcecode, prefix_method_plus_comment, prefix_method_postfix_comments, prefix_method_strengthened_comment)
            api_timeout: API request timeout in seconds (default: 90)
            api_max_retries: Maximum number of retry attempts for API calls (default: 3)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.analysis_mode = analysis_mode  # buggy_method_sourcecode, prefix_method_plus_comment, prefix_method_postfix_comments, prefix_method_strengthened_comment
        self.api_timeout = api_timeout
        self.api_max_retries = api_max_retries
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self) -> List[Dict]:
        """Load the dataset with strengthened comments"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded dataset with {len(dataset)} entries (entries with strengthened comments)")
        logger.info(f"Analysis mode: {self.analysis_mode}")
        return dataset
    
    def extract_method_without_comments(self, source_code: str) -> str:
        """Extract method source code without any comments"""
        lines = source_code.split('\n')
        cleaned_lines = []
        in_javadoc = False
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                cleaned_lines.append(line)
                continue
            
            # Handle Javadoc comments (/** ... */)
            if stripped.startswith('/**'):
                in_javadoc = True
                continue
            
            if in_javadoc and stripped.endswith('*/'):
                in_javadoc = False
                continue
            
            if in_javadoc:
                continue
            
            # Handle block comments (/* ... */) - can be single line or multi-line
            if '/*' in line and '*/' in line:
                # Single line block comment - remove the comment part
                before_comment = line[:line.find('/*')]
                after_comment = line[line.find('*/') + 2:]
                cleaned_line = before_comment + after_comment
                # Check if there are line comments after the block comment
                if '//' in cleaned_line:
                    comment_pos = cleaned_line.find('//')
                    cleaned_line = cleaned_line[:comment_pos].rstrip()
                if cleaned_line.strip():  # Only add if there's code left
                    cleaned_lines.append(cleaned_line)
                continue
            
            if '/*' in line:
                # Start of multi-line block comment
                in_block_comment = True
                before_comment = line[:line.find('/*')]
                if before_comment.strip():  # Only add if there's code before comment
                    cleaned_lines.append(before_comment)
                continue
            
            if in_block_comment and '*/' in line:
                # End of multi-line block comment
                in_block_comment = False
                after_comment = line[line.find('*/') + 2:]
                if after_comment.strip():  # Only add if there's code after comment
                    cleaned_lines.append(after_comment)
                continue
            
            if in_block_comment:
                # Skip lines inside block comment
                continue
            
            # Handle single-line comments (//) - remove comment part but keep code before it
            if '//' in line:
                comment_pos = line.find('//')
                before_comment = line[:comment_pos].rstrip()
                if before_comment:  # Only add if there's code before comment
                    cleaned_lines.append(before_comment)
                # Skip lines that are pure comments (start with //)
                continue
            
            # If we reach here, it's a regular code line
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_strengthened_comment(self, entry: Dict) -> str:
        """
        Extract strengthened comment from entry, handling both formats:
        - Format 1: entry['strengthened_comment'] (string) - from code_llama_strengthened.json
        - Format 2: entry['strengthened_comments'] (list of dicts) - from strengthened_comments_full.json
        """
        # Try Format 1: direct string field
        if 'strengthened_comment' in entry:
            sc = entry['strengthened_comment']
            if isinstance(sc, str) and sc.strip():
                return sc.strip()
        
        # Try Format 2: list of dictionaries
        if 'strengthened_comments' in entry:
            sc_list = entry['strengthened_comments']
            if isinstance(sc_list, list) and len(sc_list) > 0:
                # Get first item (usually there's only one)
                first_item = sc_list[0]
                if isinstance(first_item, dict) and 'strengthened_comment' in first_item:
                    sc = first_item['strengthened_comment']
                    if isinstance(sc, str) and sc.strip():
                        return sc.strip()
        
        # Not found in either format
        return ''
    
    def prepare_method_for_analysis(self, entry: Dict) -> str:
        """
        Prepare method code based on analysis mode.
        
        MODE 1: buggy_method_sourcecode
            - Return: Prefix source code without comments
        
        MODE 2: prefix_method_plus_comment
            - Return: Prefix comments + Prefix source code (without inline comments)
        
        MODE 3: prefix_method_postfix_comments
            - Return: Postfix comments + Prefix source code (without inline comments)
        
        MODE 4: prefix_method_strengthened_comment
            - Return: Strengthened comment + Prefix source code (without inline comments)
        """
        focal_method = entry['focal_methods'][0]
        
        # Get prefix and postfix data (allow None comments for compatibility with Dataset_for_bug_detection.json etc.)
        prefix_source = focal_method['prefix']['source_code']
        prefix_comments = focal_method['prefix'].get('comments') or ''
        postfix_comments = focal_method.get('postfix', {}).get('comments') or ''
        
        if self.analysis_mode == "buggy_method_sourcecode":
            # MODE 1: Only prefix source without any comments
            cleaned_source = self.extract_method_without_comments(prefix_source)
            return cleaned_source
        
        elif self.analysis_mode == "prefix_method_plus_comment":
            # MODE 2: Prefix comments + Prefix source without inline comments
            cleaned_source = self.extract_method_without_comments(prefix_source)
            if prefix_comments.strip():
                return f"{prefix_comments}\n{cleaned_source}"
            else:
                return cleaned_source
        
        elif self.analysis_mode == "prefix_method_postfix_comments":
            # MODE 3: Postfix comments + Prefix source without comments
            cleaned_source = self.extract_method_without_comments(prefix_source)
            if isinstance(postfix_comments, str) and postfix_comments.strip():
                return f"{postfix_comments.strip()}\n{cleaned_source}"
            else:
                return cleaned_source
        
        elif self.analysis_mode == "prefix_method_strengthened_comment":
            # MODE 4: Strengthened comment + Prefix source without comments
            cleaned_source = self.extract_method_without_comments(prefix_source)
            strengthened_comment = self.extract_strengthened_comment(entry)
            if strengthened_comment:
                return f"{strengthened_comment}\n{cleaned_source}"
            else:
                logger.warning("No strengthened_comment found in entry, using cleaned source only")
                return cleaned_source
        
        else:
            logger.error(f"Unknown analysis mode: {self.analysis_mode}")
            return prefix_source
    
    def add_line_numbers_to_method(self, method_code: str) -> str:
        """Add line numbers to method code for better LLM analysis"""
        lines = method_code.split('\n')
        numbered_lines = []
        
        # Find the first non-comment line to start numbering
        first_code_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines and comment lines at the beginning
            if stripped and not (stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('//') or stripped.endswith('*/')):
                first_code_line = i
                break
        
        # Add lines with appropriate numbering
        for i, line in enumerate(lines):
            if i < first_code_line:
                # Comment lines - no numbering
                numbered_lines.append(f"   {line}")
            else:
                # Code lines - start numbering from 1
                line_number = i - first_code_line + 1
                numbered_lines.append(f"{line_number:2d}| {line}")
        
        return '\n'.join(numbered_lines)
    
    def create_llm_prompt(self, method_code: str, mode: str) -> str:
        """Create prompt for LLM analysis based on mode"""
        
        mode_description = {
            "buggy_method_sourcecode": "without method-level comments (code only)",
            "prefix_method_plus_comment": "with original method-level comments (before bug fix)",
            "prefix_method_postfix_comments": "with updated method-level comments (after bug fix)",
            "prefix_method_strengthened_comment": "with strengthened method-level comments (Comments Strengthener)"
        }
        
        prompt = f"""Analyze the following Java method and assess its functionality.

Analysis Mode: {mode_description.get(mode, "unknown")}

Source Code (with line numbers):
{method_code}

Provide your assessment in the following JSON format:
{{
    "method_is_buggy": "Yes" or "No",
    "buggy_code_lines": "EXACT line numbers where issues occur (e.g., 'Line 5', 'Line 17', 'Lines 8-9'). If no issues, use empty string.",
    "rationale": "Technical explanation of the method's behavior and any issues found. Provide analysis without mentioning line numbers. Focus on technical aspects and implications."
}}

CRITICAL: For buggy_code_lines, you MUST provide the EXACT line numbers from the numbered code above. Do not guess or provide ranges unless the issue spans multiple consecutive lines.

Examples of correct responses:
- buggy_code_lines: "Line 17" (for a single line issue)
- buggy_code_lines: "Line 5 and Line 19" (for multiple specific line issues)
- buggy_code_lines: "Lines 8-9" (for an issue spanning consecutive lines)
- rationale: "The method calls getNode() without null checking, which can cause NullPointerException when the node is null. This violates defensive programming principles and can lead to runtime crashes." (technical explanation WITHOUT line numbers)

Respond only with valid JSON, no additional text."""
        
        return prompt
    
    def _call_api_with_retry(self, url: str, headers: Dict, data: Dict, timeout: int = 90, 
                            max_retries: int = 3, retry_delay_base: float = 2.0, 
                            api_name: str = "API") -> Optional[requests.Response]:
        """
        Call API with retry logic and exponential backoff.
        
        Args:
            url: API endpoint URL
            headers: HTTP headers
            data: Request data
            timeout: Request timeout in seconds (default: 90)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay_base: Base delay for exponential backoff in seconds (default: 2.0)
            api_name: Name of the API for logging purposes
            
        Returns:
            Response object if successful, None if all retries failed
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s, etc.
                    delay = retry_delay_base * (2 ** (attempt - 1))
                    logger.warning(f"{api_name} request failed (attempt {attempt}/{max_retries}). "
                                 f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                
                logger.debug(f"{api_name} request (attempt {attempt + 1}/{max_retries}, timeout={timeout}s)")
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
                return response
                
            except RequestsTimeout as e:
                last_exception = e
                logger.warning(f"{api_name} request timed out (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"{api_name} request timed out after {max_retries} attempts")
                    
            except (RequestsConnectionError, RequestException) as e:
                last_exception = e
                logger.warning(f"{api_name} connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"{api_name} connection failed after {max_retries} attempts")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"{api_name} unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"{api_name} request failed after {max_retries} attempts")
        
        logger.error(f"{api_name} all retry attempts failed. Last error: {last_exception}")
        return None
    
    def call_openai_gpt_api(self, prompt: str, api_key: Optional[str] = None, 
                           timeout: int = 90, max_retries: int = 3) -> Dict:
        """Call OpenAI GPT-4o API for bug analysis with retry logic"""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return {
                    "method_is_buggy": "Error",
                    "buggy_code_lines": "",
                    "rationale": "OpenAI API key not found"
                }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Java code analyzer. You must provide EXACT line numbers in buggy_code_lines field, but the rationale should NOT mention line numbers - only provide technical analysis. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # Call API with retry logic
        response = self._call_api_with_retry(
            url="https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=data,
            timeout=timeout,
            max_retries=max_retries,
            api_name="OpenAI API"
        )
        
        if response is None:
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": "OpenAI API request failed after retries (timeout or connection error)"
            }
        
        if response.status_code == 200:
            try:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    # Clean the content - remove markdown code blocks if present
                    cleaned_content = content.strip()
                    if cleaned_content.startswith('```json'):
                        cleaned_content = cleaned_content[7:]  # Remove ```json
                    if cleaned_content.startswith('```'):
                        cleaned_content = cleaned_content[3:]   # Remove ```
                    if cleaned_content.endswith('```'):
                        cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                    cleaned_content = cleaned_content.strip()
                    
                    return json.loads(cleaned_content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.error(f"LLM response content: {content}")
                    return {
                        "method_is_buggy": "Error",
                        "buggy_code_lines": "",
                        "rationale": f"LLM returned invalid JSON: {str(json_err)}"
                    }
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing OpenAI API response: {e}")
                return {
                    "method_is_buggy": "Error",
                    "buggy_code_lines": "",
                    "rationale": f"Error parsing OpenAI API response: {str(e)}"
                }
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": f"OpenAI API error: {response.status_code}"
            }
    
    def call_deepseek_api(self, prompt: str, api_key: Optional[str] = None, 
                         timeout: int = 90, max_retries: int = 3) -> Dict:
        """Call DeepSeek-Coder API for bug analysis with retry logic"""
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                logger.error("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
                return {
                    "method_is_buggy": "Error",
                    "buggy_code_lines": "",
                    "rationale": "DeepSeek API key not found"
                }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-coder",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Java code analyzer. You must provide EXACT line numbers in buggy_code_lines field, but the rationale should NOT mention line numbers - only provide technical analysis. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # Call API with retry logic
        response = self._call_api_with_retry(
            url="https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            data=data,
            timeout=timeout,
            max_retries=max_retries,
            api_name="DeepSeek-Coder API"
        )
        
        if response is None:
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": "DeepSeek-Coder API request failed after retries (timeout or connection error)"
            }
        
        if response.status_code == 200:
            try:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    # Clean the content - remove markdown code blocks if present
                    cleaned_content = content.strip()
                    if cleaned_content.startswith('```json'):
                        cleaned_content = cleaned_content[7:]  # Remove ```json
                    if cleaned_content.startswith('```'):
                        cleaned_content = cleaned_content[3:]   # Remove ```
                    if cleaned_content.endswith('```'):
                        cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                    cleaned_content = cleaned_content.strip()
                    
                    return json.loads(cleaned_content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.error(f"LLM response content: {content}")
                    return {
                        "method_is_buggy": "Error",
                        "buggy_code_lines": "",
                        "rationale": f"LLM returned invalid JSON: {str(json_err)}"
                    }
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing DeepSeek-Coder API response: {e}")
                return {
                    "method_is_buggy": "Error",
                    "buggy_code_lines": "",
                    "rationale": f"Error parsing DeepSeek-Coder API response: {str(e)}"
                }
        else:
            logger.error(f"DeepSeek-Coder API error: {response.status_code} - {response.text}")
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": f"DeepSeek-Coder API error: {response.status_code}"
            }
    
    def _require_api_key(self, model_type: str) -> None:
        """Check that the required API key is set; exit with clear message if not."""
        model_type = model_type.lower()
        if model_type == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                logger.error("OPENAI_API_KEY is not set. Set it and try again.")
                logger.info("Example (PowerShell): $env:OPENAI_API_KEY = 'your-key-here'")
                sys.exit(1)
        elif model_type == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            if not key:
                logger.error("DEEPSEEK_API_KEY is not set. Set it and try again.")
                logger.info("Example (PowerShell): $env:DEEPSEEK_API_KEY = 'your-key-here'")
                sys.exit(1)
        else:
            logger.error(f"Unknown model type: {model_type}")
            sys.exit(1)

    def call_llm_api(self, prompt: str, model_type: str = "deepseek") -> Dict:
        """Call LLM API based on model type with retry logic"""
        if model_type.lower() == "openai":
            return self.call_openai_gpt_api(prompt, timeout=self.api_timeout, max_retries=self.api_max_retries)
        elif model_type.lower() == "deepseek":
            return self.call_deepseek_api(prompt, timeout=self.api_timeout, max_retries=self.api_max_retries)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": f"Unknown model type: {model_type}"
            }
    
    def analyze_method_with_llm(self, entry: Dict, model_type: str = "deepseek") -> Dict:
        """Analyze a single method with LLM based on selected mode"""
        focal_method = entry['focal_methods'][0]
        method_name = focal_method.get('method_name', 'unknown')
        
        # Prepare method code based on analysis mode
        method_code = self.prepare_method_for_analysis(entry)
        
        # Add line numbers for better analysis
        numbered_method_code = self.add_line_numbers_to_method(method_code)
        
        logger.info(f"Analyzing method: {method_name} using {model_type} (Mode: {self.analysis_mode})")
        
        # Create prompt
        prompt = self.create_llm_prompt(numbered_method_code, self.analysis_mode)
        
        # Call LLM API
        try:
            llm_response = self.call_llm_api(prompt, model_type)
            
            # Validate response format
            if not isinstance(llm_response, dict):
                logger.error(f"Invalid response format from {model_type}")
                return {
                    "method_is_buggy": "Error",
                    "buggy_code_lines": "",
                    "rationale": f"Invalid response format from {model_type}"
                }
            
            # Ensure required fields are present
            required_fields = ["method_is_buggy", "buggy_code_lines", "rationale"]
            for field in required_fields:
                if field not in llm_response:
                    logger.warning(f"Missing field '{field}' in {model_type} response")
                    llm_response[field] = ""
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error analyzing method {method_name} with {model_type}: {e}")
            return {
                "method_is_buggy": "Error",
                "buggy_code_lines": "",
                "rationale": f"Error during analysis with {model_type}: {str(e)}"
            }
    
    def process_dataset(self, max_entries: Optional[int] = None, model_type: str = "deepseek") -> List[Dict]:
        """Process the entire pre/post-fix dataset with LLM analysis"""
        dataset = self.load_dataset()
        
        # Keep original order
        if max_entries:
            dataset = dataset[:max_entries]
            logger.info(f"Processing first {max_entries} entries using {model_type}")
        
        analyzed_dataset = []
        
        for i, entry in enumerate(dataset):
            logger.info(f"Processing entry {i+1}/{len(dataset)}")
            
            # Extract focal method
            focal_methods = entry.get('focal_methods', [])
            if not focal_methods:
                logger.warning(f"Entry {i+1} has no focal methods, skipping")
                continue
            
            # Analyze with LLM
            llm_analysis = self.analyze_method_with_llm(entry, model_type)
            
            # Add metadata
            llm_analysis["model_used"] = model_type
            llm_analysis["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            llm_analysis["analysis_mode"] = self.analysis_mode
            _pf = entry['focal_methods'][0]
            _pre_comments = (_pf['prefix'].get('comments') or '').strip()
            _post_comments = (_pf['postfix'].get('comments') or '').strip()
            llm_analysis["has_prefix_comment"] = len(_pre_comments) > 0
            llm_analysis["has_postfix_comment"] = len(_post_comments) > 0
            llm_analysis["requirements_changed"] = (_pre_comments != _post_comments)
            
            # Create new entry with LLM analysis
            new_entry = entry.copy()
            
            # Add LLM analysis to focal method
            new_entry['focal_methods'][0].update(llm_analysis)
            
            analyzed_dataset.append(new_entry)
            
            # Add delay to avoid overwhelming API
            if i % 10 == 0 and i > 0:
                time.sleep(2)
        
        return analyzed_dataset
    
    def verify_dataset_consistency(self, dataset: List[Dict]) -> None:
        """Verify that all fields are consistent across entries"""
        logger.info("Verifying dataset consistency...")
        
        issues = []
        
        for i, entry in enumerate(dataset):
            # Check required fields
            required_fields = ['commit_message', 'patch', 'buggy_program', 'focal_methods', 'bug_report']
            
            for field in required_fields:
                if field not in entry:
                    issues.append(f"Entry {i+1}: Missing field '{field}'")
            
            # Check for strengthened_comment if using MODE 4
            if self.analysis_mode == "prefix_method_strengthened_comment":
                strengthened_comment = self.extract_strengthened_comment(entry)
                if not strengthened_comment:
                    issues.append(f"Entry {i+1}: Missing or empty 'strengthened_comment'/'strengthened_comments' field (required for MODE 4)")
            
            # Check focal_methods structure (pre/post-fix format)
            focal_methods = entry.get('focal_methods', [])
            if focal_methods:
                focal_method = focal_methods[0]
                required_focal_fields = ['method_name', 'prefix', 'postfix', 
                                       'method_is_buggy', 'buggy_code_lines', 'rationale']
                
                for field in required_focal_fields:
                    if field not in focal_method:
                        issues.append(f"Entry {i+1}: Missing focal_method field '{field}'")
                
                # Check prefix/postfix structure
                if 'prefix' in focal_method:
                    if 'source_code' not in focal_method['prefix'] or 'comments' not in focal_method['prefix']:
                        issues.append(f"Entry {i+1}: Invalid prefix structure")
                
                if 'postfix' in focal_method:
                    if 'source_code' not in focal_method['postfix'] or 'comments' not in focal_method['postfix']:
                        issues.append(f"Entry {i+1}: Invalid postfix structure")
        
        if issues:
            logger.warning(f"Found {len(issues)} consistency issues:")
            for issue in issues[:10]:
                logger.warning(f"  {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")
        else:
            logger.info("[OK] All entries are consistent!")
    
    def generate_analysis_summary(self, analyzed_dataset: List[Dict]) -> None:
        """Generate comprehensive summary of LLM analysis results"""
        logger.info("\n" + "="*80)
        logger.info(f"LLM ANALYSIS SUMMARY - Mode: {self.analysis_mode}")
        logger.info("="*80)
        
        total_entries = len(analyzed_dataset)
        logger.info(f"Total entries analyzed: {total_entries}")
        
        # Count different analysis results
        buggy_count = 0
        not_buggy_count = 0
        error_count = 0
        unknown_count = 0
        
        # Track requirement evolution impact
        req_changed_buggy = 0
        req_changed_not_buggy = 0
        req_same_buggy = 0
        req_same_not_buggy = 0
        
        # Track by project
        project_stats = {}
        
        # Track common bug types
        bug_types = {}
        
        for entry in analyzed_dataset:
            focal_method = entry['focal_methods'][0]
            analysis_result = focal_method.get('method_is_buggy', 'Unknown').lower()
            req_changed = focal_method.get('requirements_changed', False)
            
            # Count by result type
            if analysis_result == 'yes':
                buggy_count += 1
                if req_changed:
                    req_changed_buggy += 1
                else:
                    req_same_buggy += 1
            elif analysis_result == 'no':
                not_buggy_count += 1
                if req_changed:
                    req_changed_not_buggy += 1
                else:
                    req_same_not_buggy += 1
            elif analysis_result == 'error':
                error_count += 1
            else:
                unknown_count += 1
            
            # Track by project
            project_name = entry.get('bug_report', {}).get('project_name', 'Unknown')
            if project_name not in project_stats:
                project_stats[project_name] = {'buggy': 0, 'not_buggy': 0, 'error': 0, 'unknown': 0, 'total': 0}
            
            project_stats[project_name]['total'] += 1
            if analysis_result == 'yes':
                project_stats[project_name]['buggy'] += 1
            elif analysis_result == 'no':
                project_stats[project_name]['not_buggy'] += 1
            elif analysis_result == 'error':
                project_stats[project_name]['error'] += 1
            else:
                project_stats[project_name]['unknown'] += 1
            
            # Track bug types from rationale
            if analysis_result == 'yes':
                rationale = focal_method.get('rationale', '').lower()
                if 'null pointer' in rationale or 'nullpointerexception' in rationale or 'npe' in rationale:
                    bug_types['NullPointerException'] = bug_types.get('NullPointerException', 0) + 1
                elif 'array' in rationale and 'bound' in rationale:
                    bug_types['ArrayIndexOutOfBoundsException'] = bug_types.get('ArrayIndexOutOfBoundsException', 0) + 1
                elif 'illegal argument' in rationale:
                    bug_types['IllegalArgumentException'] = bug_types.get('IllegalArgumentException', 0) + 1
                elif 'concurrent' in rationale or 'race condition' in rationale:
                    bug_types['ConcurrentModificationException'] = bug_types.get('ConcurrentModificationException', 0) + 1
                elif 'class cast' in rationale:
                    bug_types['ClassCastException'] = bug_types.get('ClassCastException', 0) + 1
                elif 'number format' in rationale:
                    bug_types['NumberFormatException'] = bug_types.get('NumberFormatException', 0) + 1
                elif 'illegal state' in rationale:
                    bug_types['IllegalStateException'] = bug_types.get('IllegalStateException', 0) + 1
                elif 'unsupported operation' in rationale:
                    bug_types['UnsupportedOperationException'] = bug_types.get('UnsupportedOperationException', 0) + 1
                elif 'logic error' in rationale or 'incorrect logic' in rationale or 'calculation' in rationale or 'compute' in rationale:
                    bug_types['Logic/Calculation Error'] = bug_types.get('Logic/Calculation Error', 0) + 1
                elif 'resource leak' in rationale or 'memory leak' in rationale:
                    bug_types['Resource/Memory Leak'] = bug_types.get('Resource/Memory Leak', 0) + 1
                elif 'thread safety' in rationale:
                    bug_types['Thread Safety'] = bug_types.get('Thread Safety', 0) + 1
                elif 'comment' in rationale or 'documentation' in rationale:
                    bug_types['Comment/Implementation Mismatch'] = bug_types.get('Comment/Implementation Mismatch', 0) + 1
                elif 'validation' in rationale or 'validate' in rationale or ('check' in rationale and 'missing' in rationale):
                    bug_types['Validation/Check Missing'] = bug_types.get('Validation/Check Missing', 0) + 1
                elif 'state' in rationale or 'initialization' in rationale or 'initialize' in rationale or 'uninitialized' in rationale:
                    bug_types['State/Initialization Issue'] = bug_types.get('State/Initialization Issue', 0) + 1
                elif 'return' in rationale and ('wrong' in rationale or 'incorrect' in rationale or 'missing' in rationale or 'uninitialized' in rationale):
                    bug_types['Return Value Issue'] = bug_types.get('Return Value Issue', 0) + 1
                else:
                    bug_types['Other'] = bug_types.get('Other', 0) + 1
        
        # Calculate percentages
        buggy_percentage = (buggy_count / total_entries * 100) if total_entries > 0 else 0
        not_buggy_percentage = (not_buggy_count / total_entries * 100) if total_entries > 0 else 0
        error_percentage = (error_count / total_entries * 100) if total_entries > 0 else 0
        
        # Print main statistics
        logger.info(f"\nANALYSIS RESULTS:")
        logger.info(f"  Buggy methods: {buggy_count} ({buggy_percentage:.1f}%)")
        logger.info(f"  Not buggy methods: {not_buggy_count} ({not_buggy_percentage:.1f}%)")
        logger.info(f"  Analysis errors: {error_count} ({error_percentage:.1f}%)")
        logger.info(f"  Unknown/Unclear: {unknown_count}")
        
        # Print requirement evolution impact
        logger.info(f"\nREQUIREMENT EVOLUTION IMPACT:")
        logger.info(f"  Requirements Changed & Identified as Buggy: {req_changed_buggy}")
        logger.info(f"  Requirements Changed & Identified as Not Buggy: {req_changed_not_buggy}")
        logger.info(f"  Requirements Same & Identified as Buggy: {req_same_buggy}")
        logger.info(f"  Requirements Same & Identified as Not Buggy: {req_same_not_buggy}")
        
        # Print project-wise statistics
        logger.info(f"\nPROJECT-WISE BREAKDOWN:")
        for project, stats in sorted(project_stats.items()):
            total = stats['total']
            buggy_pct = (stats['buggy'] / total * 100) if total > 0 else 0
            not_buggy_pct = (stats['not_buggy'] / total * 100) if total > 0 else 0
            logger.info(f"  {project}:")
            logger.info(f"    Total: {total} | Buggy: {stats['buggy']} ({buggy_pct:.1f}%) | Not buggy: {stats['not_buggy']} ({not_buggy_pct:.1f}%)")
            if stats['error'] > 0 or stats['unknown'] > 0:
                logger.info(f"    Errors: {stats['error']} | Unknown: {stats['unknown']}")
        
        # Print bug type distribution
        if bug_types:
            logger.info(f"\n🐛 BUG TYPE DISTRIBUTION (for {buggy_count} buggy methods):")
            sorted_bug_types = sorted(bug_types.items(), key=lambda x: x[1], reverse=True)
            for bug_type, count in sorted_bug_types:
                percentage = (count / buggy_count * 100) if buggy_count > 0 else 0
                logger.info(f"  {bug_type}: {count} ({percentage:.1f}%)")
        
        # Print key insights
        logger.info(f"\n💡 KEY INSIGHTS:")
        logger.info(f"  • Analysis Mode: {self.analysis_mode}")
        if buggy_count > not_buggy_count:
            logger.info(f"  • LLM identified MORE methods as buggy ({buggy_count}) than not buggy ({not_buggy_count})")
        elif not_buggy_count > buggy_count:
            logger.info(f"  • LLM identified MORE methods as not buggy ({not_buggy_count}) than buggy ({buggy_count})")
        else:
            logger.info(f"  • LLM identified EQUAL numbers of buggy and not buggy methods")
        
        if error_count > 0:
            logger.info(f"  • {error_count} methods had analysis errors (likely API issues)")
        
        if bug_types:
            most_common_bug = max(bug_types.items(), key=lambda x: x[1])
            logger.info(f"  • Most common bug type: {most_common_bug[0]} ({most_common_bug[1]} occurrences)")
        
        logger.info(f"\n📈 SUCCESS RATE: {((buggy_count + not_buggy_count) / total_entries * 100):.1f}% of methods were successfully analyzed")
        
        logger.info("="*80)

    def run_analysis(self, max_entries: Optional[int] = None, model_type: str = "deepseek") -> None:
        """Run the complete LLM analysis process"""
        mode_descriptions = {
            "buggy_method_sourcecode": "WITHOUT any comments (pure code analysis)",
            "prefix_method_plus_comment": "WITH original requirements (prefix comments)",
            "prefix_method_postfix_comments": "WITH updated requirements (postfix comments)",
            "prefix_method_strengthened_comment": "WITH strengthened requirements (Comments Strengthener)"
        }
        
        logger.info(f"Starting LLM bug analysis using {model_type}...")
        logger.info(f"Analysis mode: {mode_descriptions.get(self.analysis_mode, 'Unknown')}")
        
        # Fail fast if API key is missing (avoids processing all entries with errors)
        self._require_api_key(model_type)
        
        try:
            # Process dataset
            analyzed_dataset = self.process_dataset(max_entries, model_type)
            
            # Verify consistency
            self.verify_dataset_consistency(analyzed_dataset)
            
            # Save dataset with model type and mode in filename
            mode_suffix = f"_{self.analysis_mode}"
            model_suffix = f"_{model_type}"
            output_file = self.output_file.parent / f"llm_analyzed{mode_suffix}{model_suffix}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analyzed_dataset, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved analyzed dataset to: {output_file}")
            
            # Generate comprehensive summary
            self.generate_analysis_summary(analyzed_dataset)
            
            logger.info(f"[OK] LLM analysis completed successfully!")
            logger.info(f"Total entries analyzed: {len(analyzed_dataset)}")
            logger.info(f"Model used: {model_type}")
            logger.info(f"Analysis mode: {self.analysis_mode}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLM bug analysis for pre/post-fix datasets (4 modes) with optional strengthened comments."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input JSON dataset (e.g., strengthened_with_answers.json). If omitted, runs in interactive mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "buggy_method_sourcecode",
            "prefix_method_plus_comment",
            "prefix_method_postfix_comments",
            "prefix_method_strengthened_comment"
        ],
        default=None,
        help="Analysis mode. If omitted, runs in interactive mode."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "deepseek"],
        default=None,
        help="LLM provider. If omitted, runs in interactive mode."
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Max entries to process (e.g., 5). Omit to process all entries."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write output JSON files (default: current directory)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the chosen model. If set, overrides DEEPSEEK_API_KEY or OPENAI_API_KEY env var."
    )
    args = parser.parse_args()

    # If --api_key given (and we know the model), set the corresponding env var
    if args.api_key and args.model:
        if args.model == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key.strip()
        else:
            os.environ["DEEPSEEK_API_KEY"] = args.api_key.strip()

    # Non-interactive / CLI mode
    if args.input_file and args.mode and args.model:
        analyzer = GuavaPrePostFixLLMAnalyzer(
            input_file=args.input_file,
            output_file=str(Path(args.output_dir) / "llm_analyzed_prepostfix.json"),
            analysis_mode=args.mode,
        )
        analyzer.run_analysis(max_entries=args.max_entries, model_type=args.model)
        return

    print("="*80)
    print("LLM Bug Analysis - Pre/Post-Fix Requirement Evolution")
    print("="*80)
    print()
    print("This script analyzes the impact of requirements (comments) on bug detection.")
    print()
    print("Available Analysis Modes:")
    print()
    print("MODE 1: buggy_method_sourcecode")
    print("  • Input: Prefix method source code WITHOUT any comments")
    print("  • Tests: Baseline bug detection with code only")
    print()
    print("MODE 2: prefix_method_plus_comment")
    print("  • Input: Prefix method source code + Prefix method-level comments")
    print("  • Tests: Bug detection with ORIGINAL requirements (before fix)")
    print()
    print("MODE 3: prefix_method_postfix_comments")
    print("  • Input: Prefix method source code + Postfix method-level comments")
    print("  • Tests: Bug detection with UPDATED requirements (after bug fix)")
    print("  • Research Question: Do updated requirements improve bug detection?")
    print()
    print("MODE 4: prefix_method_strengthened_comment")
    print("  • Input: Prefix method source code + Strengthened comment (Comments Strengthener)")
    print("  • Tests: Bug detection with STRENGTHENED requirements (Comments Strengthener)")
    print("  • Research Question: Do strengthened requirements improve bug detection?")
    print()
    print("="*80)
    print()
    
    # Choose analysis mode
    print("Choose Analysis Mode:")
    print("1. buggy_method_sourcecode (Baseline - code only)")
    print("2. prefix_method_plus_comment (Original requirements)")
    print("3. prefix_method_postfix_comments (Updated requirements - postfix comments)")
    print("4. prefix_method_strengthened_comment (Strengthened requirements - Comments Strengthener)")
    print()
    
    mode_choice = input("Choose mode (1, 2, 3, or 4): ").strip()
    
    if mode_choice == "1":
        analysis_mode = "buggy_method_sourcecode"
        print("✓ Selected: Baseline analysis (code only, no comments)")
    elif mode_choice == "2":
        analysis_mode = "prefix_method_plus_comment"
        print("✓ Selected: Analysis with original requirements")
    elif mode_choice == "3":
        analysis_mode = "prefix_method_postfix_comments"
        print("✓ Selected: Analysis with updated requirements (postfix comments)")
    elif mode_choice == "4":
        analysis_mode = "prefix_method_strengthened_comment"
        print("✓ Selected: Analysis with strengthened requirements (Comments Strengthener)")
    else:
        print("Invalid choice. Using default: buggy_method_sourcecode")
        analysis_mode = "buggy_method_sourcecode"
    
    print()
    
    # Choose model type
    print("Choose LLM Model:")
    print("1. OpenAI GPT-4o")
    print("2. DeepSeek-Coder (Recommended for code analysis)")
    print()
    
    model_choice = input("Choose model (1 for OpenAI, 2 for DeepSeek, or press Enter for DeepSeek): ").strip()
    
    if model_choice == "1":
        model_type = "openai"
        print("[OK] Using OpenAI GPT-4o model")
    else:
        model_type = "deepseek"
        print("[OK] Using DeepSeek-Coder model (Best for Java bug analysis)")
    
    print()
    
    # Choose number of entries to process
    print("Choose number of entries to process:")
    print("1. First 5 entries (for testing)")
    print("2. First 10 entries")
    print("3. All 228 entries (entries with strengthened comments)")
    print()
    
    entries_choice = input("Choose option (1, 2, 3, or press Enter for 5): ").strip()
    
    if entries_choice == "2":
        max_entries = 10
    elif entries_choice == "3":
        max_entries = None
    else:
        max_entries = 5
    
    print(f"Processing {max_entries if max_entries else 'all'} entries...")
    print()
    
    try:
        # Create analyzer
        # Ask user which input file to use
        print("\nChoose input dataset:")
        print("1. Dataset_for_bug_detection.json (pre/post-fix; Modes 1–3 only)")
        print("2. code_llama_strengthened.json (228 entries)")
        print("3. strengthened_comments_full.json (244 entries)")
        print("4. strengthened_comments.json")
        print("5. strengthened_results.json")
        print("6. strengthened_dataset_interactive.json")
        print("7. strengthened_with_answers.json")
        print()
        
        file_choice = input("Choose file (1-7, or press Enter for Dataset_for_bug_detection.json): ").strip()
        
        if file_choice == "2":
            input_file = "code_llama_strengthened.json"
            print(f"✓ Selected: {input_file}")
        elif file_choice == "3":
            input_file = "strengthened_comments_full.json"
            print(f"✓ Selected: {input_file}")
        elif file_choice == "4":
            input_file = "strengthened_comments.json"
            print(f"✓ Selected: {input_file}")
        elif file_choice == "5":
            input_file = "strengthened_results.json"
            print(f"✓ Selected: {input_file}")
        elif file_choice == "6":
            input_file = "strengthened_dataset_interactive.json"
            print(f"✓ Selected: {input_file}")
        elif file_choice == "7":
            input_file = "strengthened_with_answers.json"
            print(f"✓ Selected: {input_file}")
        else:
            input_file = "Dataset_for_bug_detection.json"
            print(f"✓ Selected: {input_file}")
        
        analyzer = GuavaPrePostFixLLMAnalyzer(
            input_file=input_file,
            analysis_mode=analysis_mode
        )
        
        analyzer.run_analysis(max_entries=max_entries, model_type=model_type)
        
        print(f"\n[SUCCESS] LLM analysis completed!")
        print(f"Check the detailed summary above for:")
        print(f"   - Buggy vs not buggy identification rates")
        print(f"   - Impact of requirement evolution on analysis")
        print(f"   - Bug type distribution")
        print(f"   - Project-wise breakdown")
        print()
        print(f"Results saved to: llm_analyzed_{analysis_mode}_{model_type}.json")
        print()
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
