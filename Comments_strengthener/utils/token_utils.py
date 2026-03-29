"""
Token utility functions for text processing and comparison.
"""

import re
from typing import List, Set, Optional, Dict
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        pass


def extract_nouns(text: str) -> Set[str]:
    """
    Extract noun tokens from text for semantic comparison.
    
    Args:
        text: Input text
        
    Returns:
        Set of lowercase noun tokens
    """
    if not text:
        return set()
    
    try:
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Tag parts of speech
        tagged = nltk.pos_tag(tokens)
        
        # Extract nouns (NN, NNS, NNP, NNPS)
        nouns = {word for word, pos in tagged if pos.startswith('NN')}
        
        return nouns
    except LookupError:
        # Fallback: simple word extraction if NLTK resources not available
        import re
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter out common stop words and keep longer words (likely nouns)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        nouns = {w for w in words if len(w) > 3 and w not in stop_words}
        return nouns


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract all numeric tokens from text.
    
    Args:
        text: Input text
        
    Returns:
        List of numeric strings found in text
    """
    # Find all numbers (integers, floats, hex, etc.)
    pattern = r'\b\d+\.?\d*\b'
    numbers = re.findall(pattern, text)
    return numbers


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def extract_javadoc_block(text: str) -> Optional[str]:
    """
    Extract Javadoc comment block from text.
    
    Args:
        text: Text that may contain Javadoc
        
    Returns:
        Javadoc block (/** ... */) or None
    """
    pattern = r'/\*\*.*?\*/'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def extract_javadoc_tags(text: str) -> Dict[str, List[str]]:
    """
    Extract Javadoc tags from comment.
    
    Args:
        text: Javadoc comment text
        
    Returns:
        Dictionary mapping tag names to their content
    """
    tags = {}
    
    # Common Javadoc tags
    tag_patterns = {
        'param': r'@param\s+(\S+)\s+(.*?)(?=@|\*/|$)',
        'return': r'@return\s+(.*?)(?=@|\*/|$)',
        'throws': r'@throws\s+(\S+)\s+(.*?)(?=@|\*/|$)',
        'throws_alt': r'@exception\s+(\S+)\s+(.*?)(?=@|\*/|$)',
        'since': r'@since\s+(.*?)(?=@|\*/|$)',
        'deprecated': r'@deprecated\s+(.*?)(?=@|\*/|$)',
        'see': r'@see\s+(.*?)(?=@|\*/|$)',
        'author': r'@author\s+(.*?)(?=@|\*/|$)',
        'version': r'@version\s+(.*?)(?=@|\*/|$)'
    }
    
    for tag_name, pattern in tag_patterns.items():
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            tags[tag_name] = matches
    
    return tags

