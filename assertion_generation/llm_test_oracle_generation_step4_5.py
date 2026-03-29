#!/usr/bin/env python3
"""
LLM Test Oracle Generation Script for Steps 4 and 5
========================================================================

This script generates assertion statements (test oracles) for triggering test cases.

STEP 4: Generate assertions WITHOUT method-level comments
- Input: Buggy method (prefix) source code + Test prefix (without assertions)
- Task: Generate assertion statements based on buggy method and test setup
- Research Question: Can LLMs generate correct test oracles from buggy code?

STEP 5: Generate assertions WITH method-level comments
- Input: Buggy method (prefix) source code + Method-level comments + Test prefix (without assertions)
- Task: Generate assertion statements with additional context from comments
- Research Question: Do method-level comments improve oracle generation accuracy?

Dataset: strengthened_comments_full.json

Four analysis modes (--mode):
  MODE 1: step4_without_comments     -> buggy_method_sourcecode (no comments)
  MODE 2: step5_with_comments        -> prefix_method_plus_comment (original)
  MODE 3: step5_with_postfix_comments -> prefix_method_postfix_comments (fixed-version comments)
  MODE 4: step5_with_strengthened_comments -> prefix_method_strengthened_comment
"""

import json
import os
import time
import random
import copy
from pathlib import Path
from typing import Dict, List, Optional
import logging
import requests
from comment_gating_preprocessor import CommentGatingPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set fixed random seed for reproducibility
random.seed(0)

class TestOracleGenerator:
    """Generates test oracle (assertion statements) using LLM"""
    
    # Prompt variant registry - defines different prompt strategies
    PROMPT_VARIANTS = {
        "balanced": {
            "name": "Balanced (Standard)",
            "description": "Standard prompt: code + comments, warns comments may be incomplete",
            "code_weight": 0.5,
            "comment_weight": 0.5
        },
        "code_first_debias": {
            "name": "Code-First De-bias",
            "description": "Prioritize source code; treat comments as optional hints",
            "code_weight": 0.8,
            "comment_weight": 0.2
        },
        "code_only_focus": {
            "name": "Code-Only Focus",
            "description": "Strong emphasis on code analysis; minimize comment influence",
            "code_weight": 0.95,
            "comment_weight": 0.05
        },
        "comment_aware": {
            "name": "Comment-Aware",
            "description": "Actively leverage comments when present; use as primary signal",
            "code_weight": 0.3,
            "comment_weight": 0.7
        },
        "hybrid_balanced": {
            "name": "Hybrid Balanced",
            "description": "Balance code and comments; explicitly check for conflicts",
            "code_weight": 0.5,
            "comment_weight": 0.5
        },
        "trust_bias_mitigated": {
            "name": "Trust-Bias Mitigated",
            "description": "Automated comment gating: extract only Javadoc tags, treat silence as unknown",
            "code_weight": 0.7,
            "comment_weight": 0.3
        },
        "test_driven": {
            "name": "Test-Driven",
            "description": "Prioritize test prefix context; use code/comments to validate test expectations",
            "code_weight": 0.4,
            "comment_weight": 0.3,
            "test_weight": 0.3
        }
    }
    
    def __init__(self, input_file: str = "strengthened_comments_full.json", 
                 output_file: str = "llm_generated_oracles.json",
                 analysis_mode: str = "step4_without_comments",
                 prompt_style: str = "balanced"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.analysis_mode = analysis_mode  # step4_without_comments, step5_with_comments, step5_with_strengthened_comments
        # prompt_style: one of the keys in PROMPT_VARIANTS
        if prompt_style not in self.PROMPT_VARIANTS:
            logger.warning(f"Unknown prompt style '{prompt_style}', using 'balanced'")
            prompt_style = "balanced"
        self.prompt_style = prompt_style
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize comment gating preprocessor for automated filtering
        self.comment_preprocessor = CommentGatingPreprocessor()
        
    def load_dataset(self) -> List[Dict]:
        """Load the dataset with test cases without assertions"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded dataset with {len(dataset)} entries")
        logger.info(f"Analysis mode: {self.analysis_mode}")
        logger.info(f"Prompt style: {self.prompt_style}")
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
            
            # Handle block comments (/* ... */)
            if '/*' in line and '*/' in line:
                before_comment = line[:line.find('/*')]
                after_comment = line[line.find('*/') + 2:]
                cleaned_line = before_comment + after_comment
                if '//' in cleaned_line:
                    comment_pos = cleaned_line.find('//')
                    cleaned_line = cleaned_line[:comment_pos].rstrip()
                if cleaned_line.strip():
                    cleaned_lines.append(cleaned_line)
                continue
            
            if '/*' in line:
                in_block_comment = True
                before_comment = line[:line.find('/*')]
                if before_comment.strip():
                    cleaned_lines.append(before_comment)
                continue
            
            if in_block_comment and '*/' in line:
                in_block_comment = False
                after_comment = line[line.find('*/') + 2:]
                if after_comment.strip():
                    cleaned_lines.append(after_comment)
                continue
            
            if in_block_comment:
                continue
            
            # Handle single-line comments (//)
            if '//' in line:
                comment_pos = line.find('//')
                before_comment = line[:comment_pos].rstrip()
                if before_comment:
                    cleaned_lines.append(before_comment)
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def prepare_input_for_oracle_generation(self, entry: Dict) -> Dict:
        """
        Prepare input for oracle generation based on analysis mode.
        
        STEP 4: step4_without_comments
            - Buggy method (prefix) source code WITHOUT comments
            - Test prefix (without assertions)
        
        STEP 5: step5_with_comments
            - Buggy method (prefix) source code WITHOUT inline comments
            - Method-level comments (prefix comments)
            - Test prefix (without assertions)
        
        STEP 5: step5_with_postfix_comments
            - Buggy method (prefix) source code WITHOUT inline comments
            - Postfix (fixed-version) method-level comments
            - Test prefix (without assertions)
        STEP 5: step5_with_strengthened_comments
            - Buggy method (prefix) source code WITHOUT inline comments
            - Strengthened comments (from strengthened_comments field)
            - Test prefix (without assertions)
        """
        focal_method = entry['focal_methods'][0]
        
        # Get prefix data
        prefix_source = focal_method['prefix']['source_code']
        prefix_comments = focal_method['prefix']['comments']
        
        # For step5_with_postfix_comments mode, use postfix (fixed-version) comments
        if self.analysis_mode == "step5_with_postfix_comments":
            prefix_comments = focal_method.get('postfix', {}).get('comments', '') or prefix_comments
        
        # For step5_with_strengthened_comments mode, use strengthened_comments
        if self.analysis_mode == "step5_with_strengthened_comments":
            # Use strengthened_comments instead of regular prefix comments
            if 'strengthened_comments' in entry and isinstance(entry['strengthened_comments'], list):
                method_name = focal_method.get('method_name', '')
                bug_id = entry.get('bug_report', {}).get('bug_id', '')
                
                # Look for strengthened comment that matches this method
                # Prefer accepted ones, otherwise use the first available
                strengthened_comment_found = False
                for sc_entry in entry['strengthened_comments']:
                    if isinstance(sc_entry, dict) and 'strengthened_comment' in sc_entry:
                        # Check if this strengthened comment is accepted
                        if sc_entry.get('accepted', False):
                            prefix_comments = sc_entry['strengthened_comment']
                            logger.info(f"Using accepted strengthened comment for method {method_name}")
                            strengthened_comment_found = True
                            break
                
                # If no accepted one found, use the first available
                if not strengthened_comment_found:
                    for sc_entry in entry['strengthened_comments']:
                        if isinstance(sc_entry, dict) and 'strengthened_comment' in sc_entry:
                            prefix_comments = sc_entry['strengthened_comment']
                            logger.info(f"Using first available strengthened comment for method {method_name}")
                            strengthened_comment_found = True
                            break
                
                if not strengthened_comment_found:
                    logger.warning(f"No strengthened comment found for method {method_name}, falling back to regular comments")
            else:
                logger.warning(f"Entry does not have strengthened_comments field, falling back to regular comments")
        # For step5_with_comments mode, check for strengthened comments as fallback (existing behavior)
        elif self.analysis_mode == "step5_with_comments":
            # Check for strengthened comments (if using strengthened_comments_full.json)
            # strengthened_comments is a list of dicts with 'strengthened_comment' field
            if 'strengthened_comments' in entry and isinstance(entry['strengthened_comments'], list):
                # Try to find matching strengthened comment for this focal method
                method_name = focal_method.get('method_name', '')
                bug_id = entry.get('bug_report', {}).get('bug_id', '')
                
                # Look for strengthened comment that matches this method
                # Match by method_id or by checking if it's the first/only one
                for sc_entry in entry['strengthened_comments']:
                    if isinstance(sc_entry, dict) and 'strengthened_comment' in sc_entry:
                        # Check if this strengthened comment is accepted
                        if sc_entry.get('accepted', False):
                            # Use the strengthened comment instead of original
                            prefix_comments = sc_entry['strengthened_comment']
                            logger.debug(f"Using strengthened comment for method {method_name}")
                            break
                        # If no accepted one found, use the first one if available
                        elif prefix_comments == focal_method['prefix']['comments']:
                            # Only use if we haven't found an accepted one yet
                            prefix_comments = sc_entry['strengthened_comment']
                            logger.debug(f"Using first available strengthened comment for method {method_name}")
                            break
        
        # Get test prefix (without assertions)
        test_prefix = entry.get("test_case_without_assertions", "")
        if not test_prefix:
            test_prefix = entry.get("triggering_test_case", "")
        
        # Get method name and class context
        method_name = focal_method.get('method_name', 'unknown')
        buggy_program = entry.get('buggy_program', {})
        class_context = ""
        if buggy_program:
            # Get first class code for context
            first_key = list(buggy_program.keys())[0]
            class_context = buggy_program[first_key]
        
        # Prepare method code based on mode
        if self.analysis_mode == "step4_without_comments":
            # STEP 4: Method code without any comments
            method_code = self.extract_method_without_comments(prefix_source)
            method_with_comments = False
        elif self.analysis_mode in ("step5_with_comments", "step5_with_strengthened_comments", "step5_with_postfix_comments"):
            # STEP 5: Method code with method-level comments (regular or strengthened)
            cleaned_source = self.extract_method_without_comments(prefix_source)
            
            if prefix_comments.strip():
                # Use automated comment gating for trust_bias_mitigated prompt
                if self.prompt_style == "trust_bias_mitigated":
                    # Automated preprocessing: gate comments BEFORE prompt construction
                    gated_comment, comment_metadata = self.comment_preprocessor.gate_comment(prefix_comments)
                    
                    if gated_comment:
                        # Has verifiable Javadoc tags - include gated comment
                        method_code = f"{gated_comment}\n{cleaned_source}"
                        method_with_comments = True
                    else:
                        # No verifiable tags - treat as unknown, don't include comment
                        method_code = cleaned_source
                        method_with_comments = False  # Signal: COMMENT_UNKNOWN
                else:
                    # Other prompt styles: use comment (regular or strengthened)
                    method_code = f"{prefix_comments}\n{cleaned_source}"
                    method_with_comments = True
            else:
                method_code = cleaned_source
                method_with_comments = False
        else:
            logger.error(f"Unknown analysis mode: {self.analysis_mode}")
            method_code = prefix_source
            method_with_comments = False
        
        result = {
            "method_code": method_code,
            "method_name": method_name,
            "test_prefix": test_prefix,
            "class_context": class_context,
            "has_method_comments": method_with_comments,
            "prefix_comments": prefix_comments
        }
        
        # Add comment metadata for trust_bias_mitigated prompt
        if self.analysis_mode in ("step5_with_comments", "step5_with_strengthened_comments", "step5_with_postfix_comments") and self.prompt_style == "trust_bias_mitigated":
            if prefix_comments.strip():
                _, comment_metadata = self.comment_preprocessor.gate_comment(prefix_comments)
                result["comment_metadata"] = comment_metadata
            else:
                result["comment_metadata"] = {"comment_status": "EMPTY", "comment_signal": "NO_COMMENT"}
        
        # Add flags for comment source
        result["uses_strengthened_comments"] = (self.analysis_mode == "step5_with_strengthened_comments")
        result["uses_postfix_comments"] = (self.analysis_mode == "step5_with_postfix_comments")
        
        return result
    
    def _get_style_instructions(self, has_method_comments: bool) -> str:
        """Get style-specific instructions based on prompt variant"""
        
        if self.prompt_style == "code_first_debias":
            return (
                "- PRIORITIZE source code and test prefix. Comments may be incomplete or misleading.\n"
                "- Do NOT weaken or remove checks just because a comment is silent about them.\n"
                "- Include edge cases implied by code/test setup (e.g., concurrency, null handling, boundaries).\n"
                "- If comments conflict with code/test, prefer code/test behavior."
            )
        
        elif self.prompt_style == "code_only_focus":
            return (
                "- FOCUS STRONGLY on source code analysis. Code is the primary source of truth.\n"
                "- Analyze code structure, control flow, data dependencies, and edge cases from code.\n"
                "- Comments are supplementary only; do not rely on them for critical assertions.\n"
                "- Generate assertions based on what the code actually does, not what comments describe.\n"
                "- Include all edge cases visible in code (null checks, bounds, concurrency, exceptions)."
            )
        
        elif self.prompt_style == "comment_aware":
            if has_method_comments:
                return (
                    "- ACTIVELY USE method-level comments as primary source of behavioral contracts.\n"
                    "- Extract preconditions, postconditions, edge cases, and exceptions from comments.\n"
                    "- Use code to validate and supplement comment information.\n"
                    "- If comments specify behavior, generate assertions that verify that behavior.\n"
                    "- Combine comment contracts with code implementation details."
                )
            else:
                return (
                    "- Focus on source code analysis since no comments are available.\n"
                    "- Extract behavioral contracts from code structure and test context."
                )
        
        elif self.prompt_style == "hybrid_balanced":
            if has_method_comments:
                return (
                    "- BALANCE code analysis and comment interpretation equally.\n"
                    "- Extract information from both code and comments.\n"
                    "- EXPLICITLY CHECK for conflicts between code behavior and comment descriptions.\n"
                    "- If conflict exists: prefer code behavior, but note the discrepancy.\n"
                    "- Generate comprehensive assertions covering both code logic and comment contracts."
                )
            else:
                return (
                    "- Focus on source code analysis since no comments are available.\n"
                    "- Extract behavioral contracts from code structure and test context."
                )
        
        elif self.prompt_style == "test_driven":
            return (
                "- PRIORITIZE test prefix context to understand what is being tested.\n"
                "- Use test setup, method calls, and variable names to infer expected behavior.\n"
                "- Use code to validate test expectations and identify edge cases.\n"
                "- Use comments (if available) to understand intended behavior.\n"
                "- Generate assertions that verify test expectations based on test context."
            )
        
        elif self.prompt_style == "trust_bias_mitigated":
            return (
                "- AUTOMATED COMMENT GATING: Only verifiable Javadoc tags are provided (pre-filtered).\n"
                "- CRITICAL: Treat comment SILENCE as 'UNKNOWN' - do NOT infer that unmentioned behaviors are unimportant.\n"
                "- ALWAYS check ALL code-visible details (exceptions, null checks, bounds, concurrency) REGARDLESS of whether comments mention them.\n"
                "- Use Javadoc tag information to SUPPLEMENT code analysis, not replace it.\n"
                "- EXPLICIT CONFLICT RULE: If comment information contradicts code behavior, IGNORE comment and follow code.\n"
                "- Generate assertions for: (1) ALL code-visible behaviors, (2) Comment-verified behaviors when tags are present."
            )
        
        else:  # "balanced" (default)
            return (
                "- Use source code and test prefix as primary signals; comments are secondary hints.\n"
                "- Treat comments as potentially incomplete; do not ignore code/test cues such as concurrency loops or null checks.\n"
                "- Include edge cases visible from code/test even if comments do not mention them."
            )
    
    def create_oracle_generation_prompt(self, prepared_input: Dict, entry: Dict) -> str:
        """Create prompt for LLM to generate assertion statements"""
        
        method_code = prepared_input["method_code"]
        test_prefix = prepared_input["test_prefix"]
        has_method_comments = prepared_input.get("has_method_comments", False)
        
        mode_description = {
            "step4_without_comments": "WITHOUT method-level comments (buggy method source code only)",
            "step5_with_comments": "WITH prefix (original) method-level comments",
            "step5_with_postfix_comments": "WITH postfix (fixed-version) method-level comments",
            "step5_with_strengthened_comments": "WITH strengthened comments"
        }

        # Get comment metadata if available (for trust_bias_mitigated)
        comment_metadata = prepared_input.get("comment_metadata", {})
        comment_signal = comment_metadata.get("comment_signal", "NO_COMMENT")
        
        # Determine comment status message
        if self.prompt_style == "trust_bias_mitigated":
            if comment_signal == "VERIFIABLE":
                comment_status = "YES (Verifiable Javadoc tags only - pre-filtered)"
            elif comment_signal == "COMMENT_UNKNOWN":
                comment_status = "YES (Generic comment without verifiable tags - treated as UNKNOWN)"
            else:
                comment_status = "NO"
        else:
            comment_status = "YES" if has_method_comments else "NO"

        # Prompt style instructions based on variant
        style_instructions = self._get_style_instructions(has_method_comments)
        
        # Simplified prompt - ONLY the required inputs
        prompt = f"""You are a test oracle generation expert. Your task is to generate assertion statements for a test case.

**Context:**
- This is a buggy method (prefix version) that was later fixed
- The test case prefix (setup and method calls) is provided, but assertions are missing
- You need to generate appropriate assertion statements based on the buggy method's expected behavior
- Method-level comments provided: {comment_status} ({mode_description.get(self.analysis_mode, '')})

**Buggy Method {mode_description.get(self.analysis_mode, '')}:**
```java
{method_code}
```

**Test Case Prefix (without assertions):**
```java
{test_prefix}
```

**Your Task:**
Generate appropriate assertion statements that:
1. Test the expected behavior of the method based on its signature and implementation
2. Are appropriate for the test setup provided
3. Use standard JUnit assertions (assertEquals, assertTrue, assertFalse, assertNotNull, assertNull, assertThat, etc.)
4. Are syntactically correct and can be directly inserted into the test case
5. {style_instructions}

**Output Format:**
Provide ONLY the assertion statements (one or more) that should be added to the test case.
Do NOT include the test method structure, only the assertion lines.

Example output:
```java
assertEquals(expectedValue, actualValue);
assertTrue(condition);
assertNotNull(result);
```

**Important:**
- Generate assertions based on the BUGGY method's behavior (not the fixed version)
- Focus on what the method should return or how it should behave
- Use appropriate assertion types based on the return type and test context
- Do NOT include comments or explanations, only assertion statements

Generate the assertion statements now:"""
        
        return prompt
    
    def call_deepseek_api(self, prompt: str, api_key: Optional[str] = None) -> Dict:
        """Call DeepSeek-Coder API for oracle generation"""
        if not api_key:
            api_key = "sk-e17da7ffe75e432c824c3ac3c98ad3bb"  # Your DeepSeek API key
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-coder",
            "messages": [
                {
                    "role": "system",
                    "content": self._get_system_message()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract assertion statements from response
                # Remove markdown code blocks if present
                cleaned_content = content.strip()
                if '```java' in cleaned_content:
                    cleaned_content = cleaned_content.split('```java')[1].split('```')[0].strip()
                elif '```' in cleaned_content:
                    cleaned_content = cleaned_content.split('```')[1].split('```')[0].strip()
                
                return {
                    "generated_assertions": cleaned_content,
                    "raw_response": content,
                    "status": "success"
                }
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {
                    "generated_assertions": "",
                    "raw_response": "",
                    "status": f"error_{response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"DeepSeek API exception: {e}")
            return {
                "generated_assertions": "",
                "raw_response": "",
                "status": f"error_{str(e)}"
            }
    
    def call_openai_gpt_api(self, prompt: str, api_key: Optional[str] = None) -> Dict:
        """Call OpenAI GPT-4o API for oracle generation"""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return {
                    "generated_assertions": "",
                    "raw_response": "",
                    "status": "error_no_api_key"
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
                    "content": self._get_system_message()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract assertion statements
                cleaned_content = content.strip()
                if '```java' in cleaned_content:
                    cleaned_content = cleaned_content.split('```java')[1].split('```')[0].strip()
                elif '```' in cleaned_content:
                    cleaned_content = cleaned_content.split('```')[1].split('```')[0].strip()
                
                return {
                    "generated_assertions": cleaned_content,
                    "raw_response": content,
                    "status": "success"
                }
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return {
                    "generated_assertions": "",
                    "raw_response": "",
                    "status": f"error_{response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"OpenAI API exception: {e}")
            return {
                "generated_assertions": "",
                "raw_response": "",
                "status": f"error_{str(e)}"
            }
    
    def _get_system_message(self) -> str:
        """Get system message based on prompt style"""
        variant_info = self.PROMPT_VARIANTS.get(self.prompt_style, {})
        variant_name = variant_info.get("name", self.prompt_style)
        
        base_message = "You are an expert test oracle generator. Generate only assertion statements in valid Java syntax. Do not include explanations or comments, only the assertion code."
        
        if self.prompt_style == "code_first_debias":
            return f"{base_message} Prioritize code analysis over comment interpretation."
        elif self.prompt_style == "code_only_focus":
            return f"{base_message} Focus strongly on code structure and implementation details."
        elif self.prompt_style == "comment_aware":
            return f"{base_message} Actively leverage method-level comments when available."
        elif self.prompt_style == "hybrid_balanced":
            return f"{base_message} Balance code analysis and comment interpretation, checking for conflicts."
        elif self.prompt_style == "test_driven":
            return f"{base_message} Prioritize test context to understand expected behavior."
        else:
            return base_message
    
    def call_llm_api(self, prompt: str, model_type: str = "deepseek") -> Dict:
        """Call LLM API based on model type"""
        if model_type.lower() == "openai":
            return self.call_openai_gpt_api(prompt)
        elif model_type.lower() == "deepseek":
            return self.call_deepseek_api(prompt)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {
                "generated_assertions": "",
                "raw_response": "",
                "status": f"error_unknown_model"
            }
    
    def generate_oracle_for_entry(self, entry: Dict, model_type: str = "deepseek") -> Dict:
        """Generate test oracle (assertions) for a single entry"""
        focal_method = entry['focal_methods'][0]
        method_name = focal_method.get('method_name', 'unknown')
        bug_id = entry.get("bug_report", {}).get("bug_id", "unknown")[:12]
        
        logger.info(f"Generating oracle for method: {method_name} (Bug ID: {bug_id}) using {model_type}")
        
        # Prepare input
        prepared_input = self.prepare_input_for_oracle_generation(entry)
        
        # Create prompt
        prompt = self.create_oracle_generation_prompt(prepared_input, entry)
        
        # Call LLM API
        try:
            llm_response = self.call_llm_api(prompt, model_type)
            
            # Add metadata
            llm_response["model_used"] = model_type
            llm_response["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            llm_response["analysis_mode"] = self.analysis_mode
            llm_response["method_name"] = method_name
            llm_response["has_method_comments"] = prepared_input["has_method_comments"]
            llm_response["prompt_style"] = self.prompt_style
            llm_response["prompt_variant_name"] = self.PROMPT_VARIANTS.get(self.prompt_style, {}).get("name", self.prompt_style)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating oracle for {method_name}: {e}")
            return {
                "generated_assertions": "",
                "raw_response": "",
                "status": f"error_{str(e)}",
                "model_used": model_type,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_mode": self.analysis_mode,
                "method_name": method_name
            }
    
    def process_dataset(self, max_entries: Optional[int] = None, model_type: str = "deepseek") -> List[Dict]:
        """Process the entire dataset to generate test oracles"""
        dataset = self.load_dataset()
        
        if max_entries:
            dataset = dataset[:max_entries]
            logger.info(f"Processing first {max_entries} entries using {model_type}")
        
        analyzed_dataset = []
        
        for i, entry in enumerate(dataset):
            logger.info(f"Processing entry {i+1}/{len(dataset)}")
            
            # Generate oracle
            oracle_result = self.generate_oracle_for_entry(entry, model_type)
            
            # Create new entry with generated oracle - use deepcopy to preserve all nested fields
            new_entry = copy.deepcopy(entry)
            new_entry["generated_oracle"] = oracle_result
            
            # CRITICAL: Explicitly preserve strengthened_comments field if it exists in input
            # This ensures strengthened_comments_full.json dataset preserves this field in output
            if 'strengthened_comments' in entry:
                # Always ensure it's in the output, even if deepcopy somehow missed it
                if 'strengthened_comments' not in new_entry:
                    logger.warning(f"Warning: strengthened_comments field was lost during copy for entry {i+1}, restoring...")
                    new_entry['strengthened_comments'] = copy.deepcopy(entry.get('strengthened_comments'))
                else:
                    # Double-check: ensure the structure matches exactly
                    original_sc = entry.get('strengthened_comments')
                    copied_sc = new_entry.get('strengthened_comments')
                    if original_sc != copied_sc:
                        logger.warning(f"Warning: strengthened_comments structure changed for entry {i+1}, restoring...")
                        new_entry['strengthened_comments'] = copy.deepcopy(original_sc)
                # Log confirmation for debugging
                if i < 3:  # Log first 3 entries for verification
                    logger.debug(f"Entry {i+1}: strengthened_comments preserved ({len(new_entry.get('strengthened_comments', []))} items)")
            
            # Final verification: ensure ALL original fields are preserved
            # This is a critical check to ensure nothing is lost, especially strengthened_comments
            for key in entry.keys():
                if key not in new_entry:
                    logger.warning(f"Warning: Field '{key}' was lost for entry {i+1}, restoring...")
                    new_entry[key] = copy.deepcopy(entry[key])
                elif key == 'strengthened_comments':
                    # Extra check for strengthened_comments to ensure it's exactly preserved
                    if entry[key] != new_entry[key]:
                        logger.warning(f"Warning: strengthened_comments structure differs for entry {i+1}, restoring...")
                        new_entry[key] = copy.deepcopy(entry[key])
            
            analyzed_dataset.append(new_entry)
            
            # Add delay to avoid overwhelming API
            if i % 10 == 0 and i > 0:
                time.sleep(2)
        
        return analyzed_dataset
    
    def generate_summary(self, analyzed_dataset: List[Dict]) -> None:
        """Generate summary of oracle generation results"""
        logger.info("\n" + "="*80)
        logger.info(f"TEST ORACLE GENERATION SUMMARY - Mode: {self.analysis_mode}")
        logger.info("="*80)
        
        total_entries = len(analyzed_dataset)
        successful_generations = sum(1 for e in analyzed_dataset 
                                    if e.get("generated_oracle", {}).get("status") == "success")
        failed_generations = total_entries - successful_generations
        
        logger.info(f"Total entries: {total_entries}")
        logger.info(f"Successful oracle generations: {successful_generations} ({successful_generations*100/total_entries:.1f}%)")
        logger.info(f"Failed generations: {failed_generations} ({failed_generations*100/total_entries:.1f}%)")
        
        # Count entries with assertions generated
        entries_with_assertions = sum(1 for e in analyzed_dataset 
                                     if e.get("generated_oracle", {}).get("generated_assertions", "").strip())
        
        logger.info(f"Entries with generated assertions: {entries_with_assertions} ({entries_with_assertions*100/total_entries:.1f}%)")
        
        logger.info("="*80)
    
    def run_analysis(self, max_entries: Optional[int] = None, model_type: str = "deepseek") -> None:
        """Run the complete oracle generation process"""
        mode_descriptions = {
            "step4_without_comments": "MODE 1: Buggy method source code only",
            "step5_with_comments": "MODE 2: Prefix method + original comment",
            "step5_with_postfix_comments": "MODE 3: Prefix method + postfix comments",
            "step5_with_strengthened_comments": "MODE 4: Prefix method + strengthened comment"
        }
        
        variant_info = self.PROMPT_VARIANTS.get(self.prompt_style, {})
        variant_name = variant_info.get("name", self.prompt_style)
        
        logger.info(f"Starting test oracle generation using {model_type}...")
        logger.info(f"Analysis mode: {mode_descriptions.get(self.analysis_mode, 'Unknown')}")
        logger.info(f"Prompt style: {variant_name} ({self.prompt_style})")
        
        try:
            # Process dataset
            analyzed_dataset = self.process_dataset(max_entries, model_type)
            
            # Save dataset
            mode_suffix = f"_{self.analysis_mode}"
            model_suffix = f"_{model_type}"
            prompt_suffix = f"_{self.prompt_style}"
            output_file = self.output_file.parent / f"llm_generated_oracles{mode_suffix}{model_suffix}{prompt_suffix}.json"
            
            # Final verification before saving: ensure all input fields are preserved
            logger.info("Performing final verification of output data before saving...")
            dataset = self.load_dataset()
            if max_entries:
                dataset = dataset[:max_entries]
            for idx, output_entry in enumerate(analyzed_dataset):
                if idx < len(dataset):
                    input_entry = dataset[idx]
                    # Verify all original fields are present
                    for key in input_entry.keys():
                        if key not in output_entry:
                            logger.warning(f"Final check: Field '{key}' missing in output entry {idx+1}, restoring...")
                            output_entry[key] = copy.deepcopy(input_entry[key])
                        elif key == 'strengthened_comments' and input_entry.get(key) != output_entry.get(key):
                            logger.warning(f"Final check: strengthened_comments differs in entry {idx+1}, restoring...")
                            output_entry[key] = copy.deepcopy(input_entry[key])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analyzed_dataset, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved generated oracles to: {output_file}")
            
            # Generate summary
            self.generate_summary(analyzed_dataset)
            
            logger.info(f"[OK] Oracle generation completed successfully!")
            logger.info(f"Total entries processed: {len(analyzed_dataset)}")
            logger.info(f"Model used: {model_type}")
            logger.info(f"Analysis mode: {self.analysis_mode}")
            logger.info(f"Prompt style: {variant_name}")
            
        except Exception as e:
            logger.error(f"Oracle generation failed: {e}")
            raise
    
    @classmethod
    def run_batch_analysis(cls, input_file: str, analysis_mode: str, 
                          prompt_variants: List[str], max_entries: Optional[int] = None,
                          model_type: str = "deepseek") -> Dict[str, List[Dict]]:
        """
        Run analysis with multiple prompt variants for comparison.
        
        Args:
            input_file: Path to input dataset
            analysis_mode: step4_without_comments, step5_with_comments, step5_with_postfix_comments, or step5_with_strengthened_comments
            prompt_variants: List of prompt style keys to test
            max_entries: Maximum number of entries to process (None for all)
            model_type: LLM model to use
            
        Returns:
            Dictionary mapping prompt_style -> list of analyzed entries
        """
        results = {}
        
        logger.info("="*80)
        logger.info(f"BATCH PROMPT VARIANT ANALYSIS")
        logger.info("="*80)
        logger.info(f"Analysis mode: {analysis_mode}")
        logger.info(f"Prompt variants to test: {prompt_variants}")
        logger.info(f"Model: {model_type}")
        logger.info(f"Max entries: {max_entries if max_entries else 'all'}")
        logger.info("="*80)
        
        for i, variant in enumerate(prompt_variants, 1):
            if variant not in cls.PROMPT_VARIANTS:
                logger.warning(f"Skipping unknown prompt variant: {variant}")
                continue
            
            variant_info = cls.PROMPT_VARIANTS[variant]
            logger.info(f"\n[{i}/{len(prompt_variants)}] Testing prompt variant: {variant_info['name']}")
            logger.info(f"Description: {variant_info['description']}")
            
            generator = cls(
                input_file=input_file,
                analysis_mode=analysis_mode,
                prompt_style=variant
            )
            
            try:
                analyzed_dataset = generator.process_dataset(max_entries, model_type)
                results[variant] = analyzed_dataset
                
                # Final verification before saving: ensure all input fields are preserved
                logger.info(f"Performing final verification for variant '{variant}' before saving...")
                dataset = generator.load_dataset()
                if max_entries:
                    dataset = dataset[:max_entries]
                for idx, output_entry in enumerate(analyzed_dataset):
                    if idx < len(dataset):
                        input_entry = dataset[idx]
                        # Verify all original fields are present
                        for key in input_entry.keys():
                            if key not in output_entry:
                                logger.warning(f"Final check [{variant}]: Field '{key}' missing in output entry {idx+1}, restoring...")
                                output_entry[key] = copy.deepcopy(input_entry[key])
                            elif key == 'strengthened_comments' and input_entry.get(key) != output_entry.get(key):
                                logger.warning(f"Final check [{variant}]: strengthened_comments differs in entry {idx+1}, restoring...")
                                output_entry[key] = copy.deepcopy(input_entry[key])
                
                # Save individual result
                mode_suffix = f"_{analysis_mode}"
                model_suffix = f"_{model_type}"
                prompt_suffix = f"_{variant}"
                output_file = generator.output_file.parent / f"llm_generated_oracles{mode_suffix}{model_suffix}{prompt_suffix}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(analyzed_dataset, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Saved results to: {output_file}")
                
                # Brief summary
                successful = sum(1 for e in analyzed_dataset 
                               if e.get("generated_oracle", {}).get("status") == "success")
                logger.info(f"✓ Success rate: {successful}/{len(analyzed_dataset)} ({successful*100/len(analyzed_dataset):.1f}%)")
                
            except Exception as e:
                logger.error(f"✗ Failed to process variant {variant}: {e}")
                results[variant] = []
        
        # Save combined results for comparison
        comparison_file = Path(input_file).parent / f"prompt_variant_comparison_{analysis_mode}_{model_type}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n[OK] Batch analysis completed!")
        logger.info(f"Comparison results saved to: {comparison_file}")
        logger.info(f"Tested {len(results)} prompt variants")
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Test Oracle Generation - Steps 4 and 5")
    parser.add_argument("--input_file", type=str, default=None, help="Input JSON file path (default: strengthened_comments_full.json)")
    parser.add_argument("--mode", type=str, choices=["step4_without_comments", "step5_with_comments", "step5_with_postfix_comments", "step5_with_strengthened_comments"], default=None, help="Analysis mode: 1=buggy_sourcecode, 2=prefix+comment, 3=prefix+postfix_comments, 4=prefix+strengthened_comment")
    parser.add_argument("--prompt_style", type=str, default=None, help="Prompt style variant")
    parser.add_argument("--model", type=str, choices=["openai", "deepseek"], default=None, help="LLM model to use")
    parser.add_argument("--max_entries", type=int, default=None, help="Maximum number of entries to process")
    parser.add_argument("--batch", action="store_true", help="Run batch analysis for all prompt variants")
    
    args = parser.parse_args()
    
    # If command-line arguments provided, use them; otherwise use interactive mode
    use_cli = args.input_file is not None or args.mode is not None or args.prompt_style is not None
    
    if use_cli:
        # Command-line mode
        input_file = args.input_file if args.input_file else "strengthened_comments_full.json"
        analysis_mode = args.mode if args.mode else "step5_with_comments"
        prompt_style = args.prompt_style if args.prompt_style else "hybrid_balanced"
        model_type = args.model if args.model else "deepseek"
        max_entries = args.max_entries
        
        print("="*80)
        print("LLM Test Oracle Generation - Command Line Mode")
        print("="*80)
        print(f"Input file: {input_file}")
        print(f"Analysis mode: {analysis_mode}")
        print(f"Prompt style: {prompt_style}")
        print(f"Model: {model_type}")
        print(f"Max entries: {max_entries if max_entries else 'all'}")
        print("="*80)
        print()
        
        # Verify input file exists and has strengthened_comments if expected
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"[ERROR] Input file not found: {input_file}")
            return
        
        # Check if input file has strengthened_comments
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            if len(sample_data) > 0:
                has_sc = 'strengthened_comments' in sample_data[0]
                print(f"Input file check: 'strengthened_comments' field present: {has_sc}")
                if has_sc:
                    sc_count = len(sample_data[0].get('strengthened_comments', []))
                    print(f"  First entry has {sc_count} strengthened comment(s)")
        except Exception as e:
            print(f"Warning: Could not verify input file structure: {e}")
        
        print()
        
        try:
            if args.batch:
                # Batch mode
                prompt_variants = list(TestOracleGenerator.PROMPT_VARIANTS.keys())
                TestOracleGenerator.run_batch_analysis(
                    input_file=input_file,
                    analysis_mode=analysis_mode,
                    prompt_variants=prompt_variants,
                    max_entries=max_entries,
                    model_type=model_type
                )
            else:
                # Single variant mode
                generator = TestOracleGenerator(
                    input_file=input_file,
                    analysis_mode=analysis_mode,
                    prompt_style=prompt_style
                )
                generator.run_analysis(max_entries=max_entries, model_type=model_type)
            
            print(f"\n[SUCCESS] Oracle generation completed!")
            print("="*80)
        except Exception as e:
            print(f"[ERROR] Oracle generation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        return
    
    # Interactive mode (original code)
    print("="*80)
    print("LLM Test Oracle Generation - Steps 4 and 5")
    print("Prompt Variant Testing (Per Supervisor Instructions)")
    print("="*80)
    print()
    print("This script generates assertion statements (test oracles) for test cases.")
    print()
    print("STEP 4: Generate assertions WITHOUT method-level comments")
    print("  • Input: Buggy method (prefix) + Test prefix (without assertions)")
    print("  • Task: Generate assertion statements based on buggy method")
    print()
    print("STEP 5: Generate assertions WITH method-level comments")
    print("  • Input: Buggy method (prefix) + Method comments + Test prefix")
    print("  • Task: Generate assertion statements with comment context")
    print("  • Research Question: Do comments improve oracle generation?")
    print()
    print("PROMPT VARIANTS: Test different prompt strategies to maximize benefit")
    print("  and minimize negative impact of comments")
    print()
    print("="*80)
    print()
    
    # Choose analysis mode (four modes)
    print("Choose Analysis Mode:")
    print("1. MODE 1: Buggy method source code only (no comments)")
    print("2. MODE 2: Prefix method + original (prefix) comment")
    print("3. MODE 3: Prefix method + postfix (fixed-version) comments")
    print("4. MODE 4: Prefix method + strengthened comment")
    print()
    
    mode_choice = input("Choose mode (1, 2, 3, or 4): ").strip()
    
    if mode_choice == "1":
        analysis_mode = "step4_without_comments"
        print("[OK] Selected: MODE 1 - Buggy method source code only")
    elif mode_choice == "2":
        analysis_mode = "step5_with_comments"
        print("[OK] Selected: MODE 2 - Prefix method + original comment")
    elif mode_choice == "3":
        analysis_mode = "step5_with_postfix_comments"
        print("[OK] Selected: MODE 3 - Prefix method + postfix comments")
    elif mode_choice == "4":
        analysis_mode = "step5_with_strengthened_comments"
        print("[OK] Selected: MODE 4 - Prefix method + strengthened comment")
    else:
        print("Invalid choice. Using default: MODE 1")
        analysis_mode = "step4_without_comments"
    
    print()
    
    # Choose model type
    print("Choose LLM Model:")
    print("1. OpenAI GPT-4o")
    print("2. DeepSeek-Coder (Recommended)")
    print()
    
    model_choice = input("Choose model (1 for OpenAI, 2 for DeepSeek, or press Enter for DeepSeek): ").strip()
    
    if model_choice == "1":
        model_type = "openai"
        print("[OK] Using OpenAI GPT-4o model")
    else:
        model_type = "deepseek"
        print("[OK] Using DeepSeek-Coder model")
    
    print()
    
    # Choose prompt variant mode
    print("Choose Prompt Variant Mode:")
    print("1. Single prompt variant (standard)")
    print("2. Batch test multiple prompt variants (for comparison)")
    print()
    
    variant_mode = input("Choose mode (1 or 2, or press Enter for 1): ").strip()
    
    if variant_mode == "2":
        # Batch mode: test multiple variants
        print("\nAvailable Prompt Variants:")
        for i, (key, info) in enumerate(TestOracleGenerator.PROMPT_VARIANTS.items(), 1):
            print(f"  {i}. {key}: {info['name']}")
            print(f"     {info['description']}")
        
        print()
        print("Enter prompt variant keys to test (comma-separated, e.g., balanced,code_first_debias,code_only_focus):")
        print("Or enter numbers (e.g., 1,2,5) to select by number")
        print("Or press Enter to test all variants")
        
        variants_input = input("Variants: ").strip()
        
        if variants_input:
            # Create mapping from number to variant key
            variant_list = list(TestOracleGenerator.PROMPT_VARIANTS.keys())
            number_to_key = {str(i+1): key for i, key in enumerate(variant_list)}
            
            # Parse input
            input_parts = [v.strip() for v in variants_input.split(',')]
            prompt_variants = []
            
            for part in input_parts:
                # Check if it's a number
                if part.isdigit() and part in number_to_key:
                    prompt_variants.append(number_to_key[part])
                # Check if it's a valid variant key
                elif part in TestOracleGenerator.PROMPT_VARIANTS:
                    prompt_variants.append(part)
                else:
                    print(f"Warning: '{part}' is not a valid variant number or key. Skipping.")
            
            if not prompt_variants:
                print("No valid variants selected. Using all variants.")
                prompt_variants = list(TestOracleGenerator.PROMPT_VARIANTS.keys())
        else:
            prompt_variants = list(TestOracleGenerator.PROMPT_VARIANTS.keys())
        
        print(f"\n✓ Will test {len(prompt_variants)} prompt variants: {', '.join(prompt_variants)}")
        
        # Choose number of entries
        print("\nChoose number of entries to process:")
        print("1. First 2 entries (for testing)")
        print("2. First 10 entries")
        print("3. All 244 entries")
        
        entries_choice = input("Choose option (1, 2, 3, or press Enter for 2): ").strip()
        
        if entries_choice == "2":
            max_entries = 10
        elif entries_choice == "3":
            max_entries = None
        else:
            max_entries = 2
        
        print(f"\nProcessing {max_entries if max_entries else 'all 244'} entries with {len(prompt_variants)} variants...")
        print()
        
        try:
            results = TestOracleGenerator.run_batch_analysis(
                input_file="strengthened_comments_full.json",
                analysis_mode=analysis_mode,
                prompt_variants=prompt_variants,
                max_entries=max_entries,
                model_type=model_type
            )
            
            print(f"\n[SUCCESS] Batch prompt variant analysis completed!")
            print(f"Individual results saved with suffix: _{analysis_mode}_{model_type}_<variant>")
            print(f"Comparison file: prompt_variant_comparison_{analysis_mode}_{model_type}.json")
            print()
            print("="*80)
            
        except Exception as e:
            print(f"[ERROR] Batch analysis failed: {e}")
            print("Please check the error and try again.")
    
    else:
        # Single variant mode
        print("\nAvailable Prompt Variants:")
        for i, (key, info) in enumerate(TestOracleGenerator.PROMPT_VARIANTS.items(), 1):
            print(f"  {i}. {key}: {info['name']}")
            print(f"     {info['description']}")
        
        print()
        variant_choice = input("Enter prompt variant key or number (or press Enter for 'balanced'): ").strip()
        
        if variant_choice:
            # Create mapping from number to variant key
            variant_list = list(TestOracleGenerator.PROMPT_VARIANTS.keys())
            number_to_key = {str(i+1): key for i, key in enumerate(variant_list)}
            
            # Check if it's a number
            if variant_choice.isdigit() and variant_choice in number_to_key:
                prompt_style = number_to_key[variant_choice]
            # Check if it's a valid variant key
            elif variant_choice in TestOracleGenerator.PROMPT_VARIANTS:
                prompt_style = variant_choice
            else:
                print(f"Warning: '{variant_choice}' is not valid. Using 'balanced'.")
                prompt_style = "balanced"
        else:
            prompt_style = "balanced"
        
        variant_info = TestOracleGenerator.PROMPT_VARIANTS[prompt_style]
        print(f"✓ Selected: {variant_info['name']}")
        print()
        
        # Choose number of entries to process
        print("Choose number of entries to process:")
        print("1. First 2 entries (for testing)")
        print("2. First 10 entries")
        print("3. All 244 entries")
        print()
        
        entries_choice = input("Choose option (1, 2, 3, or press Enter for 2): ").strip()
        
        if entries_choice == "2":
            max_entries = 10
        elif entries_choice == "3":
            max_entries = None
        else:
            max_entries = 2
        
        print(f"Processing {max_entries if max_entries else 'all 244'} entries...")
        print()
        
        try:
            # Create generator
            generator = TestOracleGenerator(
                input_file="strengthened_comments_full.json",
                analysis_mode=analysis_mode,
                prompt_style=prompt_style
            )
            
            generator.run_analysis(max_entries=max_entries, model_type=model_type)
            
            print(f"\n[SUCCESS] Oracle generation completed!")
            print(f"Results saved to: llm_generated_oracles_{analysis_mode}_{model_type}_{prompt_style}.json")
            print()
            print("="*80)
            
        except Exception as e:
            print(f"[ERROR] Oracle generation failed: {e}")
            print("Please check the error and try again.")

if __name__ == "__main__":
    main()

