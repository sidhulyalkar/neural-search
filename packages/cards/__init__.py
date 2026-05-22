"""
Neural Search Cards Package

Generate dataset cards with metadata, readiness scores, and analysis suggestions.
"""

from .generator import CardGenerator
from .markdown import MarkdownRenderer

__all__ = ["CardGenerator", "MarkdownRenderer"]
