"""
Gap Detection Module

Detects contract-relevant gaps in comments and generates questions for developers.
"""

from gap_detector.models import Gap, Question
from gap_detector.detector import GapDetector
from gap_detector.question_generator import QuestionGenerator
from gap_detector.routing import GapRouter
from gap_detector.question_bank import QuestionBank

__all__ = ['Gap', 'Question', 'GapDetector', 'QuestionGenerator', 'GapRouter', 'QuestionBank']

