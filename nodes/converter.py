"""
Document conversion node - converts RTF, plain text, and rich text to markdown.
"""

import re
from pathlib import Path

from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ..state import AudiobookState, WorkflowStage


def detect_file_type(file_path: str) -> str:
    """Detect the file type based on extension and content."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".rtf":
        return "rtf"
    elif suffix == ".md":
        return "markdown"
    elif suffix == ".txt":
        return "text"
    elif suffix in [".html", ".htm"]:
        return "html"
    elif suffix == ".docx":
        return "docx"
    else:
        # Try to detect from content
        with open(file_path, "r", errors="ignore") as f:
            start = f.read(100)
            if start.startswith("{\\rtf"):
                return "rtf"
            elif start.startswith("<!DOCTYPE") or start.startswith("<html"):
                return "html"
            else:
                return "text"


def convert_rtf_to_markdown(file_path: str) -> tuple[str, dict]:
    """Convert RTF file to markdown."""
    with open(file_path, "r", errors="ignore") as f:
        rtf_content = f.read()

    # Extract metadata from RTF if present
    metadata = {}
    title_match = re.search(r"\\title\s+([^}]+)", rtf_content)
    author_match = re.search(r"\\author\s+([^}]+)", rtf_content)

    if title_match:
        metadata["title"] = title_match.group(1).strip()
    if author_match:
        metadata["author"] = author_match.group(1).strip()

    # Convert RTF to plain text
    text = rtf_to_text(rtf_content)

    # Clean up the text
    text = clean_rtf_artifacts(text)

    # Convert to markdown format
    markdown = convert_text_to_markdown(text)

    return markdown, metadata


def convert_html_to_markdown(file_path: str) -> tuple[str, dict]:
    """Convert HTML file to markdown."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "lxml")

    # Extract metadata
    metadata = {}
    title_tag = soup.find("title")
    if title_tag:
        metadata["title"] = title_tag.get_text().strip()

    author_meta = soup.find("meta", attrs={"name": "author"})
    if author_meta:
        metadata["author"] = author_meta.get("content", "").strip()

    # Convert to markdown
    markdown = md(html_content, heading_style="ATX")

    return markdown, metadata


def convert_docx_to_markdown(file_path: str) -> tuple[str, dict]:
    """Convert DOCX file to markdown."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")

    doc = Document(file_path)
    metadata = {}

    # Extract metadata
    if doc.core_properties.title:
        metadata["title"] = doc.core_properties.title
    if doc.core_properties.author:
        metadata["author"] = doc.core_properties.author

    # Convert paragraphs to markdown
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue

        # Check if it's a heading based on style
        style_name = para.style.name.lower() if para.style else ""
        if "heading 1" in style_name:
            lines.append(f"# {text}")
        elif "heading 2" in style_name:
            lines.append(f"## {text}")
        elif "heading 3" in style_name:
            lines.append(f"### {text}")
        elif "title" in style_name:
            lines.append(f"# {text}")
        else:
            lines.append(text)

        lines.append("")

    markdown = "\n".join(lines)
    return markdown, metadata


def convert_text_to_markdown(text: str) -> str:
    """
    Convert plain text to markdown format.
    Attempts to detect structure like chapters, paragraphs, etc.
    """
    lines = text.split("\n")
    result = []
    in_paragraph = False

    for i, line in enumerate(lines):
        line = line.rstrip()

        # Skip empty lines but preserve paragraph breaks
        if not line:
            if in_paragraph:
                result.append("")
                in_paragraph = False
            continue

        # Detect chapter headings
        chapter_patterns = [
            r"^(CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)\s*[:\-]?\s*(.*)$",
            r"^(PART|Part)\s+(\d+|[IVXLCDM]+)\s*[:\-]?\s*(.*)$",
            r"^(\d+)\.\s+(.+)$",  # Numbered chapters like "1. Introduction"
            r"^([IVXLCDM]+)\.\s+(.+)$",  # Roman numeral chapters
        ]

        is_heading = False
        for pattern in chapter_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # This looks like a chapter heading
                result.append(f"\n## {line}\n")
                is_heading = True
                in_paragraph = False
                break

        if is_heading:
            continue

        # Check if line is ALL CAPS (might be a heading)
        if line.isupper() and len(line) > 3 and len(line) < 100:
            # Likely a heading
            result.append(f"\n## {line.title()}\n")
            in_paragraph = False
            continue

        # Regular paragraph text
        result.append(line)
        in_paragraph = True

    return "\n".join(result)


def clean_rtf_artifacts(text: str) -> str:
    """Clean up common RTF conversion artifacts."""
    # Remove multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Remove multiple newlines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page break artifacts
    text = re.sub(r"\f", "\n\n", text)

    # Remove common RTF artifacts
    text = re.sub(r"\\[a-z]+\d*\s?", "", text)

    # Clean up smart quotes and other special chars
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...",  # Ellipsis
        "\xa0": " ",  # Non-breaking space
        "\u00a0": " ",  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def convert_document(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Convert input document to markdown.

    Args:
        state: Current workflow state

    Returns:
        Updated state with markdown_content populated
    """
    state.stage = WorkflowStage.CONVERTING

    file_path = state.input_file
    if not Path(file_path).exists():
        state.errors.append(f"Input file not found: {file_path}")
        state.stage = WorkflowStage.FAILED
        return state

    try:
        file_type = detect_file_type(file_path)

        # Read raw content for reference
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        state.raw_content = raw_bytes.decode("utf-8", errors="ignore")

        # Convert based on file type
        if file_type == "rtf":
            markdown, metadata = convert_rtf_to_markdown(file_path)
        elif file_type == "html":
            markdown, metadata = convert_html_to_markdown(file_path)
        elif file_type == "docx":
            markdown, metadata = convert_docx_to_markdown(file_path)
        elif file_type == "markdown":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                markdown = f.read()
            metadata = {}
        else:  # plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            markdown = convert_text_to_markdown(text)
            metadata = {}

        state.markdown_content = markdown
        state.book_title = metadata.get("title")
        state.book_author = metadata.get("author")

        # Move to QA stage
        state.stage = WorkflowStage.CONVERSION_QA

    except Exception as e:
        state.errors.append(f"Conversion error: {str(e)}")
        state.stage = WorkflowStage.FAILED

    return state


def get_conversion_stats(state: AudiobookState) -> dict:
    """Get statistics about the conversion for QA."""
    if not state.markdown_content:
        return {"error": "No markdown content"}

    content = state.markdown_content
    lines = content.split("\n")
    words = len(content.split())

    # Count headings
    h1_count = len(re.findall(r"^#\s", content, re.MULTILINE))
    h2_count = len(re.findall(r"^##\s", content, re.MULTILINE))
    h3_count = len(re.findall(r"^###\s", content, re.MULTILINE))

    # Count paragraphs
    paragraphs = [p for p in content.split("\n\n") if p.strip()]

    return {
        "total_characters": len(content),
        "total_words": words,
        "total_lines": len(lines),
        "total_paragraphs": len(paragraphs),
        "h1_headings": h1_count,
        "h2_headings": h2_count,
        "h3_headings": h3_count,
        "has_title": state.book_title is not None,
        "has_author": state.book_author is not None,
    }
