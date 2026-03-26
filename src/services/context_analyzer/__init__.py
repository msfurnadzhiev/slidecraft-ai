"""Production content analysis for RetrievedContext.

Analyses retrieved chunks to extract thematic clusters, score relevance,
categorise content by document role, and discover inter-chunk relationships.
The resulting ContextAnalysis feeds directly into PresentationPlanner.
"""

from .analyzer import ContentAnalyzer

__all__ = ["ContentAnalyzer"]
