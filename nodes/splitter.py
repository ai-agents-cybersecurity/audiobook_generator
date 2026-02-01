"""
Chapter splitting node - detects and splits book content into chapters.

Handles books with Part/Chapter structure (like Accelerando) by:
1. Detecting Part headers as section markers
2. Detecting named chapters under parts
3. Merging Part intro text into first chapter of that part
"""

import re
from typing import Optional

from ..state import AudiobookState, Chapter, WorkflowStage


# Pattern for Part headers (Part 1, Part I, etc.)
PART_PATTERN = r"^(?:#+\s*)?(PART|Part)\s+(\d+|[IVXLCDM]+)\s*[:\.\-]?\s*(.*)$"

# Patterns for actual chapters (named chapters that follow parts)
CHAPTER_PATTERNS = [
    # Standard chapter with number: "Chapter 1: Title" or "## Chapter 1: Title"
    r"^(?:#+\s*)?(CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)\s*[:\.\-]?\s*(.*)$",

    # Named chapter (single word title, often used in literary fiction)
    # e.g., "## Lobsters" or "Lobsters" (as heading)
    r"^##\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$",

    # ALL CAPS chapter title (standalone line)
    r"^([A-Z][A-Z]+(?:\s+[A-Z]+)*)$",
]

# Patterns to EXCLUDE (appendix, index, etc.)
EXCLUDE_PATTERNS = [
    r"(?i)^(contents|table of contents|index|appendix|glossary|notes|bibliography|references)$",
    r"(?i)^(how you got here|things you should|where to go for|what to do).*$",
    r"(?i)^(dedication|acknowledgements?|about the author|copyright)$",
]


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result = 0
    prev = 0
    for char in reversed(roman.upper()):
        curr = values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def parse_number(num_str: str) -> int:
    """Parse a number string (digit or roman numeral) to int."""
    num_str = num_str.strip()
    if num_str.isdigit():
        return int(num_str)
    elif all(c in "IVXLCDM" for c in num_str.upper()):
        return roman_to_int(num_str)
    return -1


def is_excluded_heading(text: str) -> bool:
    """Check if a heading should be excluded (appendix, index, etc.)."""
    text_clean = text.strip().lstrip("#").strip()
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    return False


def find_structural_elements(content: str) -> list[dict]:
    """
    Find all structural elements (Parts and Chapters) in the content.

    Returns:
        List of dicts with keys: type ('part' or 'chapter'), number, title, line_index
    """
    lines = content.split("\n")
    elements = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Skip excluded headings
        if is_excluded_heading(line_stripped):
            continue

        # Check for Part header
        part_match = re.match(PART_PATTERN, line_stripped, re.IGNORECASE)
        if part_match:
            part_num = parse_number(part_match.group(2))
            part_title = part_match.group(3).strip() if part_match.group(3) else ""
            elements.append({
                "type": "part",
                "number": part_num,
                "title": part_title or f"Part {part_num}",
                "line_index": i,
            })
            continue

        # Check for Chapter patterns
        for pattern in CHAPTER_PATTERNS:
            chapter_match = re.match(pattern, line_stripped, re.IGNORECASE)
            if chapter_match:
                groups = [g for g in chapter_match.groups() if g]

                # Determine chapter number and title based on pattern
                if "CHAPTER" in pattern.upper() or "chapter" in pattern.lower():
                    # Pattern with explicit "Chapter X: Title"
                    chapter_num = parse_number(groups[1]) if len(groups) > 1 else -1
                    chapter_title = groups[2].strip() if len(groups) > 2 else groups[1].strip()
                else:
                    # Named chapter pattern (just title)
                    chapter_num = -1  # Will be assigned sequentially
                    chapter_title = groups[0].strip()

                # Skip very short titles or single letters
                if len(chapter_title) < 3:
                    continue

                elements.append({
                    "type": "chapter",
                    "number": chapter_num,
                    "title": chapter_title,
                    "line_index": i,
                })
                break

    return elements


def build_chapters_from_elements(content: str, elements: list[dict]) -> list[Chapter]:
    """
    Build Chapter objects from structural elements.

    Handles Part/Chapter structure by:
    - Assigning sequential chapter numbers
    - Including Part name in chapter title
    - Merging short Part intros into following chapter
    """
    if not elements:
        return []

    lines = content.split("\n")
    chapters = []
    chapter_counter = 1
    current_part = None

    # Filter to get only chapters (Parts are used as context)
    chapter_elements = []
    for i, elem in enumerate(elements):
        if elem["type"] == "part":
            current_part = elem
        elif elem["type"] == "chapter":
            chapter_elements.append({
                **elem,
                "part": current_part,
            })
            current_part = None  # Reset after first chapter uses it

    # Now build chapters from chapter elements
    for i, elem in enumerate(chapter_elements):
        # Determine start and end lines
        start_line = elem["line_index"]

        # Find end line (start of next element or end of content)
        if i + 1 < len(chapter_elements):
            end_line = chapter_elements[i + 1]["line_index"]
        else:
            end_line = len(lines)

        # If there's a Part before this chapter, include it
        if elem.get("part"):
            part = elem["part"]
            # Check if Part header comes right before this chapter
            part_line = part["line_index"]
            if part_line < start_line:
                start_line = part_line

        # Extract content
        chapter_content = "\n".join(lines[start_line:end_line]).strip()

        if not chapter_content or len(chapter_content) < 100:
            continue

        # Build title
        if elem.get("part"):
            title = f"{elem['part']['title']}: {elem['title']}"
        else:
            title = elem["title"]

        # Assign chapter number
        if elem["number"] > 0:
            chapter_num = elem["number"]
        else:
            chapter_num = chapter_counter

        chapters.append(Chapter(
            number=chapter_num,
            title=title,
            content=chapter_content
        ))
        chapter_counter = chapter_num + 1

    return chapters


def detect_front_matter(content: str) -> tuple[Optional[str], str]:
    """
    Detect and separate front matter (title page, copyright, etc.) from main content.

    Returns:
        (front_matter, main_content)
    """
    lines = content.split("\n")

    # Look for first Part or Chapter indicator
    first_structural_line = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check for Part
        if re.match(PART_PATTERN, line_stripped, re.IGNORECASE):
            first_structural_line = i
            break

        # Check for Chapter
        for pattern in CHAPTER_PATTERNS:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                first_structural_line = i
                break
        if first_structural_line is not None:
            break

    if first_structural_line is not None and first_structural_line > 10:
        # There's substantial content before the first chapter
        front_matter = "\n".join(lines[:first_structural_line]).strip()
        main_content = "\n".join(lines[first_structural_line:]).strip()
        return front_matter, main_content

    return None, content


def split_by_size(content: str, target_chapters: int = 10) -> list[Chapter]:
    """
    Fallback: split content into roughly equal chapters by size.
    Used when no chapter markers are detected.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return [Chapter(number=1, title="Chapter 1", content=content)]

    # Calculate paragraphs per chapter
    paras_per_chapter = max(1, len(paragraphs) // target_chapters)

    chapters = []
    current_paras = []
    chapter_num = 1

    for i, para in enumerate(paragraphs):
        current_paras.append(para)

        if len(current_paras) >= paras_per_chapter and chapter_num < target_chapters:
            chapters.append(Chapter(
                number=chapter_num,
                title=f"Chapter {chapter_num}",
                content="\n\n".join(current_paras)
            ))
            current_paras = []
            chapter_num += 1

    # Add remaining paragraphs
    if current_paras:
        chapters.append(Chapter(
            number=chapter_num,
            title=f"Chapter {chapter_num}",
            content="\n\n".join(current_paras)
        ))

    return chapters


def split_chapters(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Split document into chapters.

    Args:
        state: Current workflow state with markdown_content

    Returns:
        Updated state with chapters list populated
    """
    state.stage = WorkflowStage.SPLITTING

    if not state.markdown_content:
        state.errors.append("No markdown content to split")
        state.stage = WorkflowStage.FAILED
        return state

    try:
        content = state.markdown_content
        print(f"[DEBUG SPLIT] markdown_content length: {len(content)}")

        # Detect and separate front matter
        front_matter, main_content = detect_front_matter(content)
        print(f"[DEBUG SPLIT] front_matter: {len(front_matter) if front_matter else 0}, main_content: {len(main_content)}")

        # Find structural elements (Parts and Chapters)
        elements = find_structural_elements(main_content)
        print(f"[DEBUG SPLIT] Found {len(elements)} structural elements:")
        for elem in elements:
            print(f"[DEBUG SPLIT]   {elem['type'].upper()} {elem['number']}: {elem['title'][:40]} at line {elem['line_index']}")

        # Build chapters from elements
        if elements:
            chapters = build_chapters_from_elements(main_content, elements)
        else:
            # Fallback: split by size
            state.warnings.append("No chapter markers detected, splitting by content size")
            chapters = split_by_size(main_content)

        # Add front matter as chapter 0 if present and substantial
        if front_matter and len(front_matter) > 500:
            chapters.insert(0, Chapter(
                number=0,
                title="Front Matter",
                content=front_matter
            ))

        print(f"[DEBUG SPLIT] Created {len(chapters)} final chapters:")
        for ch in chapters:
            print(f"[DEBUG SPLIT]   Chapter {ch.number}: '{ch.title[:50]}' - {len(ch.content)} chars")

        state.chapters = chapters
        state.stage = WorkflowStage.SPLIT_QA

    except Exception as e:
        import traceback
        print(f"[DEBUG SPLIT] Error: {e}")
        print(traceback.format_exc())
        state.errors.append(f"Chapter splitting error: {str(e)}")
        state.stage = WorkflowStage.FAILED

    return state


def get_split_stats(state: AudiobookState) -> dict:
    """Get statistics about chapter split for QA."""
    if not state.chapters:
        return {"error": "No chapters"}

    chapter_sizes = [len(ch.content) for ch in state.chapters]

    return {
        "total_chapters": len(state.chapters),
        "avg_chapter_size": sum(chapter_sizes) // len(chapter_sizes),
        "min_chapter_size": min(chapter_sizes),
        "max_chapter_size": max(chapter_sizes),
        "chapter_titles": [ch.title for ch in state.chapters],
        "has_front_matter": state.chapters[0].number == 0 if state.chapters else False,
    }
