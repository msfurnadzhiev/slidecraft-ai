"""Production content analysis for RetrievalContext.

Analyses retrieved passages to extract thematic clusters, score relevance,
categorise content by document role, and discover inter-passage relationships.
The resulting AnalyzedContent feeds directly into PresentationPlanner.
"""

from .analyzer import ContentAnalyzer

__all__ = ["ContentAnalyzer"]
