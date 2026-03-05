"""Shared text utilities for the PEX/SAQ pipeline."""

import re

# Footer patterns found in the enrichment corpus
_FOOTER_PATTERNS = [
    re.compile(r"\n?Feedback welcome at\n?ketaminenightmares@gmail\.com\s*$"),
    re.compile(r"\n?Special thanks to [^\n]+\.\s*$"),
]

# Source paths to skip (duplicates from HTML save artifacts)
SKIP_PATHS = frozenset([
    "pharmacology/analgesics/2000A15_opioids_respiratory_effects_files/2000A15_opioids_respiratory_effects.htm",
])

# Shared activation threshold for content dimension scores
DIMENSION_THRESHOLD = 0.6


def strip_footer(text: str) -> str:
    """Remove boilerplate footer text that contaminates classifier results."""
    for pattern in _FOOTER_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()
