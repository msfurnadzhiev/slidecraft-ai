"""Constants and category rules for content analysis."""

import re
from typing import Dict, Set

from app.schemas.analysis import ContentCategory

# Category classification rules for content categorisation
CATEGORY_KEYWORDS: Dict[ContentCategory, Set[str]] = {
    ContentCategory.INTRODUCTION: {
        "introduction", "overview", "purpose", "objective", "scope",
        "aim", "abstract", "presents", "proposes", "outline",
    },
    ContentCategory.BACKGROUND: {
        "background", "context", "literature", "previous", "prior",
        "existing", "related", "review", "studies", "research",
        "theoretical", "framework", "foundation",
    },
    ContentCategory.DATA: {
        "data", "results", "figure", "table", "chart", "graph",
        "findings", "measured", "observed", "experiment", "sample",
        "statistics", "values", "percentage", "dataset", "metric",
    },
    ContentCategory.ANALYSIS: {
        "analysis", "discussion", "comparison", "implication",
        "suggests", "indicates", "demonstrates", "interpretation",
        "evaluation", "assessment", "correlation", "trend",
    },
    ContentCategory.KEY_FINDING: {
        "significant", "important", "key", "critical", "notably",
        "remarkably", "novel", "breakthrough", "major", "primary",
        "essential", "noteworthy", "substantial",
    },
    ContentCategory.CONCLUSION: {
        "conclusion", "summary", "recommendation", "future",
        "overall", "finally", "summarize", "closing", "remarks",
    },
}

# Categories that are more likely to appear on early pages
EARLY_CATEGORIES: Set[ContentCategory] = {
    ContentCategory.INTRODUCTION,
    ContentCategory.BACKGROUND,
}

# Categories that are more likely to appear on late pages
LATE_CATEGORIES: Set[ContentCategory] = {ContentCategory.CONCLUSION}

# Categories that are more likely to contain numeric data
NUMERIC_BONUS_CATEGORIES: Set[ContentCategory] = {ContentCategory.DATA}

# Categories that are more likely to be retrieved from the document
RETRIEVAL_BONUS_CATEGORIES: Set[ContentCategory] = {ContentCategory.KEY_FINDING}

# Pattern to match numbers in the text
NUMBER_PATTERN = re.compile(r"\d+\.?\d*%?")

# Minimum number of passages required for clustering
MIN_PASSAGES_FOR_CLUSTERING = 3

# Scoring weights for category classification
W_KEYWORD = 3.0
W_POSITION = 1.5
W_NUMERIC = 1.0
W_RETRIEVAL = 1.5
