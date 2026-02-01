"""
Text chunking node - splits text into chunks suitable for TTS generation.

Uses semantic-aware chunking that respects:
- Sentence boundaries
- Paragraph structure
- Natural speech pauses
- TTS context limits (~500 characters recommended for natural speech)
"""

import re
from typing import Optional

from ..state import AudiobookState, WorkflowStage


# Sentence-ending punctuation
SENTENCE_ENDINGS = ".!?"

# Strong break points (paragraph/section boundaries)
STRONG_BREAKS = ["\n\n", "\n"]

# Medium break points (sentence boundaries)
MEDIUM_BREAKS = [".", "!", "?", ";"]

# Weak break points (clause boundaries)
WEAK_BREAKS = [",", ":", " - ", " — "]


def find_sentence_boundaries(text: str) -> list[int]:
    """
    Find all sentence boundary positions in text.

    Returns:
        List of positions where sentences end
    """
    boundaries = []

    # Match sentence endings followed by space and capital letter or end of text
    # Using a simpler pattern without variable-width lookbehind
    pattern = r'[.!?]["\')\]]*\s+(?=[A-Z"])|[.!?]["\')\]]*$'

    for match in re.finditer(pattern, text):
        boundaries.append(match.end())

    return boundaries


def find_paragraph_boundaries(text: str) -> list[int]:
    """Find paragraph boundary positions."""
    boundaries = []

    for match in re.finditer(r"\n\n+", text):
        boundaries.append(match.start())

    return boundaries


def find_best_split_point(text: str, target_pos: int, tolerance: int = 100) -> int:
    """
    Find the best position to split text near target_pos.

    Prefers (in order):
    1. Paragraph boundaries
    2. Sentence boundaries
    3. Clause boundaries
    4. Word boundaries

    Args:
        text: The text to split
        target_pos: Target position for the split
        tolerance: How far from target_pos we can search

    Returns:
        Best split position
    """
    search_start = max(0, target_pos - tolerance)
    search_end = min(len(text), target_pos + tolerance)
    search_text = text[search_start:search_end]

    # Look for paragraph break
    para_match = re.search(r"\n\n", search_text)
    if para_match:
        return search_start + para_match.end()

    # Look for sentence end
    sentence_match = re.search(r'[.!?]["\')\]]*\s+', search_text)
    if sentence_match:
        return search_start + sentence_match.end()

    # Look for clause boundary
    clause_match = re.search(r"[,;:]\s+", search_text)
    if clause_match:
        return search_start + clause_match.end()

    # Fall back to word boundary
    word_match = re.search(r"\s+", search_text)
    if word_match:
        return search_start + word_match.end()

    # Last resort: split at target
    return target_pos


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    # Using a simpler approach that doesn't require variable-width lookbehind
    # First, mark split points with a unique delimiter
    marked = re.sub(r'([.!?])\s+([A-Z])', r'\1|||SPLIT|||\2', text)

    # Split on the delimiter
    sentences = marked.split('|||SPLIT|||')
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text: str, max_chunk_size: int = 500, min_chunk_size: int = 100) -> list[str]:
    """
    Split text into chunks respecting sentence boundaries.

    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters before considering a chunk complete

    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If single sentence exceeds max, we need to split it
        if sentence_len > max_chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split long sentence at clause boundaries
            sub_chunks = split_long_sentence(sentence, max_chunk_size)
            chunks.extend(sub_chunks)
            continue

        # Check if adding this sentence exceeds limit
        if current_size + sentence_len + 1 > max_chunk_size:
            # Only flush if we have meaningful content
            if current_size >= min_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_len + 1  # +1 for space

    # Add remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_long_sentence(sentence: str, max_size: int) -> list[str]:
    """
    Split a long sentence at clause boundaries.

    Args:
        sentence: Sentence to split
        max_size: Maximum chunk size

    Returns:
        List of chunks
    """
    if len(sentence) <= max_size:
        return [sentence]

    chunks = []
    current_pos = 0

    while current_pos < len(sentence):
        if len(sentence) - current_pos <= max_size:
            # Remaining text fits
            chunks.append(sentence[current_pos:].strip())
            break

        # Find best split point
        target = current_pos + max_size
        split_pos = find_best_split_point(sentence, target, tolerance=max_size // 4)

        # Ensure we make progress
        if split_pos <= current_pos:
            split_pos = min(current_pos + max_size, len(sentence))

        chunk = sentence[current_pos:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        current_pos = split_pos

    return chunks


def chunk_chapter(content: str, max_chunk_size: int = 500) -> list[str]:
    """
    Chunk a chapter's content into TTS-appropriate segments.

    Args:
        content: Chapter content
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of chunks
    """
    if not content:
        return []

    # First, split by paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    all_chunks = []

    for para in paragraphs:
        if not para:
            continue

        # Chunk the paragraph
        para_chunks = chunk_by_sentences(para, max_chunk_size)
        all_chunks.extend(para_chunks)

    # Post-process: merge very short chunks with neighbors
    merged_chunks = merge_short_chunks(all_chunks, min_size=100, max_size=max_chunk_size)

    return merged_chunks


def merge_short_chunks(chunks: list[str], min_size: int = 100, max_size: int = 500) -> list[str]:
    """
    Merge very short chunks with neighboring chunks.

    Args:
        chunks: List of chunks
        min_size: Minimum chunk size before considering merge
        max_size: Maximum allowed chunk size

    Returns:
        List of merged chunks
    """
    if not chunks:
        return []

    merged = []
    buffer = ""

    for chunk in chunks:
        if not chunk:
            continue

        if not buffer:
            buffer = chunk
        elif len(buffer) < min_size and len(buffer) + len(chunk) + 1 <= max_size:
            # Merge short chunk with previous
            buffer = buffer + " " + chunk
        elif len(chunk) < min_size and len(buffer) + len(chunk) + 1 <= max_size:
            # Merge short chunk with previous
            buffer = buffer + " " + chunk
        else:
            # Flush buffer and start new
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def chunk_text(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Chunk cleaned chapter text for TTS.

    Args:
        state: Current workflow state with cleaned chapters

    Returns:
        Updated state with chunks in each chapter
    """
    state.stage = WorkflowStage.CHUNKING

    if not state.chapters:
        state.errors.append("No chapters to chunk")
        state.stage = WorkflowStage.FAILED
        return state

    try:
        total_chunks = 0

        print(f"[DEBUG] Chunking {len(state.chapters)} chapters with chunk_size={state.chunk_size}")

        for chapter in state.chapters:
            content = chapter.cleaned_content or chapter.content

            cleaned_len = len(chapter.cleaned_content) if chapter.cleaned_content else 0
            content_len = len(chapter.content) if chapter.content else 0
            print(f"[DEBUG] Chapter {chapter.number} '{chapter.title[:30]}...': cleaned_content={cleaned_len}, content={content_len}")

            if not content:
                print(f"[DEBUG] Chapter {chapter.number}: NO CONTENT - both cleaned_content and content are empty!")
                chapter.chunks = []
                continue

            # Debug: show first 200 chars of content being chunked
            print(f"[DEBUG] Chapter {chapter.number} content preview: {repr(content[:200])}")

            chunks = chunk_chapter(content, max_chunk_size=state.chunk_size)
            chapter.chunks = chunks
            total_chunks += len(chunks)

            if len(chunks) == 0 and len(content) > 0:
                print(f"[DEBUG] WARNING: Chapter {chapter.number} has {len(content)} chars but produced 0 chunks!")
                # Try to understand why - check paragraph splits
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                print(f"[DEBUG]   Paragraphs found: {len(paragraphs)}")
                if paragraphs:
                    print(f"[DEBUG]   First paragraph ({len(paragraphs[0])} chars): {repr(paragraphs[0][:100])}")
            else:
                print(f"[DEBUG] Chapter {chapter.number}: {len(chunks)} chunks created")

        state.total_chunks = total_chunks
        print(f"[DEBUG] Total chunks across all chapters: {total_chunks}")
        state.stage = WorkflowStage.GENERATING

    except Exception as e:
        import traceback
        print(f"[DEBUG] Chunking error: {e}")
        print(traceback.format_exc())
        state.errors.append(f"Chunking error: {str(e)}")
        state.stage = WorkflowStage.FAILED

    return state


def get_chunking_stats(state: AudiobookState) -> dict:
    """Get statistics about chunking."""
    if not state.chapters:
        return {"error": "No chapters"}

    all_chunk_sizes = []
    chapter_stats = []

    for ch in state.chapters:
        if ch.chunks:
            sizes = [len(c) for c in ch.chunks]
            all_chunk_sizes.extend(sizes)
            chapter_stats.append({
                "chapter": ch.number,
                "num_chunks": len(ch.chunks),
                "avg_chunk_size": sum(sizes) // len(sizes) if sizes else 0,
            })

    return {
        "total_chunks": len(all_chunk_sizes),
        "avg_chunk_size": sum(all_chunk_sizes) // len(all_chunk_sizes) if all_chunk_sizes else 0,
        "min_chunk_size": min(all_chunk_sizes) if all_chunk_sizes else 0,
        "max_chunk_size": max(all_chunk_sizes) if all_chunk_sizes else 0,
        "chapter_stats": chapter_stats,
    }
