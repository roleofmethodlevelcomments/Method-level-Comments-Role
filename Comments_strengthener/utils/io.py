"""
Input/Output utilities for dataset processing.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON dataset file
        
    Returns:
        List of dataset entries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save strengthened comment results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def extract_method_data(entry: Dict[str, Any], use_prefix: bool = True) -> List[Dict[str, Any]]:
    """
    Extract method data from dataset entry.
    
    Args:
        entry: Dataset entry dictionary
        use_prefix: If True, use prefix (buggy) version, else use postfix (fixed)
        
    Returns:
        List of method data dictionaries with method_id, method_code, original_comment
    """
    import copy
    methods = []
    
    for idx, focal_method in enumerate(entry.get('focal_methods', [])):
        version = 'prefix' if use_prefix else 'postfix'
        
        method_id = f"{entry.get('bug_report', {}).get('bug_id', 'unknown')}_{focal_method.get('method_name', 'unknown')}_{idx}"
        
        # CRITICAL: When use_prefix=True, strip all postfix data from entry_data
        # to ensure system only uses prefix data (experimental constraint)
        if use_prefix:
            # Create a sanitized entry_data with postfix data removed
            sanitized_entry = copy.deepcopy(entry)
            # Remove postfix from all focal_methods
            for fm in sanitized_entry.get('focal_methods', []):
                if 'postfix' in fm:
                    del fm['postfix']
            entry_data = sanitized_entry
        else:
            # When using postfix, keep full entry (for completeness)
            entry_data = entry
        
        method_data = {
            'method_id': method_id,
            'method_code': focal_method[version]['source_code'],
            'original_comment': focal_method[version]['comments'],
            'entry_data': entry_data,  # Sanitized: no postfix data when use_prefix=True
            'requirements_changed': focal_method.get('requirements_changed', False),
        }
        
        methods.append(method_data)
    
    return methods

