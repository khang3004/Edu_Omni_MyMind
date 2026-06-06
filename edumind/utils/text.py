"""EduMIND Utils — Text Processing and Chunking Utilities.

Provides layout-aware section header detection and sentence-boundary-aware
text splitting for ingestion workflows.
"""

from __future__ import annotations

import re


def is_section_header(line: str) -> bool:
    """Determines if a given line structurally qualifies as a section header.

    Rules:
      1. Short lines (< 100 chars) written in ALL CAPS.
      2. Lines ending with a colon ':'.
      3. Lines starting with numbered indexes (e.g., '1.', '2.1', 'Chương 1').

    Args:
        line: Text line string to inspect.

    Returns:
        True if the line is identified as a heading.
    """
    if not line or len(line) < 2:
        return False

    # ALL CAPS check for short lines
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars and all(c.isupper() for c in alpha_chars) and len(line) < 100:
        return True

    # Heading ends with colon
    if line.endswith(":") and len(line) < 100:
        return True

    # Numbered index patterns (e.g., "1. Introduction", "2.1 Background", "Chapter 3")
    if re.match(r"^\d+(\.\d+)*\.?\s", line) or re.match(
        r"^(Chapter|Chương|Bài|Phần)\s", line, re.IGNORECASE
    ):
        return True

    return False


def split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    """Slices a long text into overlapping chunks cleanly at sentence/word boundaries.

    Args:
        text: Large paragraph text to slice.
        max_size: Maximum character length allowed per slice.
        overlap: Slicing character overlap size.

    Returns:
        A list of text strings representing the sub-chunks.
    """
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_size

        # Attempt to split gracefully at sentence or word boundaries
        if end < len(text):
            best_break = text.rfind(". ", start, end)
            if best_break == -1 or best_break <= start:
                best_break = text.rfind(" ", start, end)
            if best_break > start:
                end = best_break + 1

        parts.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end

    return [p for p in parts if p]
