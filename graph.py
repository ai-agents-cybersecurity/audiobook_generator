"""
LangGraph workflow for audiobook generation.

This module defines the complete StateGraph workflow with:
- Document conversion
- Chapter splitting
- Text cleaning
- Audio generation
- QA verification with conditional edges for retry
"""

from typing import Literal

from langgraph.graph import StateGraph, END
from rich.console import Console

from .state import AudiobookState, WorkflowStage
from .nodes.converter import convert_document
from .nodes.splitter import split_chapters
from .nodes.cleaner import clean_text
from .nodes.chunker import chunk_text
from .nodes.tts_preprocessor import preprocess_tts
from .nodes.tts import generate_audio
from .nodes.verifier import verify_audio
from .qa_agents.conversion_qa import ConversionQAAgent
from .qa_agents.chapter_qa import ChapterQAAgent
from .qa_agents.audio_qa import AudioQAAgent

console = Console()


# Initialize QA agents
conversion_qa = ConversionQAAgent()
chapter_qa = ChapterQAAgent()
audio_qa = AudioQAAgent()


# --- Node Wrapper Functions ---

def _save_checkpoint(state: AudiobookState, step_name: str):
    """Save checkpoint after successful step."""
    try:
        checkpoint_path = state.save_checkpoint()
        console.print(f"[dim]Checkpoint saved: {step_name}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save checkpoint: {e}[/yellow]")


def node_convert(state: AudiobookState) -> AudiobookState:
    """Convert document to markdown."""
    console.print("\n[bold blue]Step 1: Converting document...[/bold blue]")
    state = convert_document(state)
    _save_checkpoint(state, "conversion")
    return state


def node_conversion_qa(state: AudiobookState) -> AudiobookState:
    """Run conversion QA."""
    state = conversion_qa.validate(state)
    if state.conversion_qa_passed:
        _save_checkpoint(state, "conversion_qa_passed")
    return state


def node_split(state: AudiobookState) -> AudiobookState:
    """Split into chapters."""
    console.print("\n[bold blue]Step 2: Splitting into chapters...[/bold blue]")
    state = split_chapters(state)
    _save_checkpoint(state, "split")
    return state


def node_split_qa(state: AudiobookState) -> AudiobookState:
    """Run chapter split QA."""
    state = chapter_qa.validate(state)
    if state.split_qa_passed:
        _save_checkpoint(state, "split_qa_passed")
    return state


def node_clean(state: AudiobookState) -> AudiobookState:
    """Clean text for TTS."""
    console.print("\n[bold blue]Step 3: Cleaning text for TTS...[/bold blue]")
    state = clean_text(state)
    _save_checkpoint(state, "clean")
    return state


def node_chunk(state: AudiobookState) -> AudiobookState:
    """Chunk text for TTS."""
    console.print("\n[bold blue]Step 4: Chunking text...[/bold blue]")
    state = chunk_text(state)
    _save_checkpoint(state, "chunk")
    return state


def node_preprocess_tts(state: AudiobookState) -> AudiobookState:
    """Preprocess chunks with Qwen3-TTS tags."""
    console.print("\n[bold blue]Step 5: Preprocessing text for Qwen3-TTS...[/bold blue]")
    state = preprocess_tts(state)
    _save_checkpoint(state, "preprocess_tts")
    return state


def node_generate(state: AudiobookState) -> AudiobookState:
    """Generate audio."""
    console.print("\n[bold blue]Step 6: Generating audio...[/bold blue]")
    state = generate_audio(state)
    _save_checkpoint(state, "generate")
    return state


def node_verify(state: AudiobookState) -> AudiobookState:
    """Verify audio."""
    console.print("\n[bold blue]Step 7: Verifying audio...[/bold blue]")
    state = verify_audio(state)
    _save_checkpoint(state, "verify")
    return state


def node_audio_qa(state: AudiobookState) -> AudiobookState:
    """Run audio QA."""
    state = audio_qa.validate(state)
    _save_checkpoint(state, "audio_qa")
    return state


def node_complete(state: AudiobookState) -> AudiobookState:
    """Mark workflow complete."""
    state.stage = WorkflowStage.COMPLETE
    console.print("\n[bold green]Audiobook generation complete![/bold green]")

    # Print summary
    chapters_done = sum(1 for ch in state.chapters if ch.audio_file)
    console.print(f"\nSummary:")
    console.print(f"  - Chapters generated: {chapters_done}/{len(state.chapters)}")
    console.print(f"  - Output directory: {state.output_dir}")

    if state.chapters:
        for ch in state.chapters:
            if ch.audio_file:
                console.print(f"    ✓ {ch.audio_file}")

    return state


def node_failed(state: AudiobookState) -> AudiobookState:
    """Handle workflow failure."""
    state.stage = WorkflowStage.FAILED
    console.print("\n[bold red]Audiobook generation failed![/bold red]")

    if state.errors:
        console.print("\nErrors:")
        for error in state.errors:
            console.print(f"  - {error}")

    return state


# --- Conditional Edge Functions ---

def route_after_conversion_qa(state: AudiobookState) -> Literal["split", "convert", "failed"]:
    """Route after conversion QA."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"

    if state.conversion_qa_passed:
        return "split"

    # Retry if under max attempts
    if state.conversion_qa_attempts < state.max_qa_attempts:
        console.print(f"[yellow]Retrying conversion (attempt {state.conversion_qa_attempts + 1}/{state.max_qa_attempts})...[/yellow]")
        return "convert"

    # Accept with warnings if issues are minor
    console.print("[yellow]Proceeding despite conversion issues (max retries reached)[/yellow]")
    return "split"


def route_after_split_qa(state: AudiobookState) -> Literal["clean", "split", "failed"]:
    """Route after split QA."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"

    if state.split_qa_passed:
        return "clean"

    # Retry if under max attempts
    if state.split_qa_attempts < state.max_qa_attempts:
        console.print(f"[yellow]Retrying split (attempt {state.split_qa_attempts + 1}/{state.max_qa_attempts})...[/yellow]")
        return "split"

    # Accept with warnings
    console.print("[yellow]Proceeding despite split issues (max retries reached)[/yellow]")
    return "clean"


def route_after_audio_qa(state: AudiobookState) -> Literal["complete", "generate", "failed"]:
    """Route after audio QA."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"

    if state.audio_qa_passed:
        return "complete"

    # Check if we should retry
    if state.audio_qa_attempts < state.max_qa_attempts:
        # Only retry if there are specific chapters to regenerate
        if state.audio_qa_issues:
            console.print(f"[yellow]Some issues found, but proceeding to completion...[/yellow]")

    return "complete"


def route_after_clean(state: AudiobookState) -> Literal["chunk", "failed"]:
    """Route after clean step."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"
    return "chunk"


def route_after_chunk(state: AudiobookState) -> Literal["preprocess_tts", "failed"]:
    """Route after chunk step."""
    if state.stage == WorkflowStage.FAILED:
        console.print("[red]Chunking failed - stopping workflow[/red]")
        return "failed"
    if state.total_chunks == 0:
        console.print("[red]No chunks created - stopping workflow[/red]")
        state.errors.append("No chunks created from chapters")
        state.stage = WorkflowStage.FAILED
        return "failed"
    return "preprocess_tts"


def route_after_preprocess_tts(state: AudiobookState) -> Literal["generate", "failed"]:
    """Route after TTS preprocessing step."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"
    return "generate"


def route_after_generate(state: AudiobookState) -> Literal["verify", "failed"]:
    """Route after generate step."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"
    return "verify"


def route_after_verify(state: AudiobookState) -> Literal["audio_qa", "failed"]:
    """Route after verify step."""
    if state.stage == WorkflowStage.FAILED:
        return "failed"
    return "audio_qa"


# --- Build the Graph ---

def build_audiobook_graph() -> StateGraph:
    """
    Build the LangGraph workflow for audiobook generation.

    The workflow follows this structure:

    convert -> conversion_qa -> [split or retry or fail]
       |
       v
    split -> split_qa -> [clean or retry or fail]
       |
       v
    clean -> chunk -> preprocess_tts -> generate -> verify -> audio_qa -> [complete or retry or fail]
       |
       v
    complete (or failed)

    Returns:
        Compiled StateGraph
    """
    # Create the graph with our state type
    workflow = StateGraph(AudiobookState)

    # Add nodes
    workflow.add_node("convert", node_convert)
    workflow.add_node("conversion_qa", node_conversion_qa)
    workflow.add_node("split", node_split)
    workflow.add_node("split_qa", node_split_qa)
    workflow.add_node("clean", node_clean)
    workflow.add_node("chunk", node_chunk)
    workflow.add_node("preprocess_tts", node_preprocess_tts)
    workflow.add_node("generate", node_generate)
    workflow.add_node("verify", node_verify)
    workflow.add_node("audio_qa", node_audio_qa)
    workflow.add_node("complete", node_complete)
    workflow.add_node("failed", node_failed)

    # Set entry point
    workflow.set_entry_point("convert")

    # Add edges

    # Convert -> Conversion QA
    workflow.add_edge("convert", "conversion_qa")

    # Conversion QA -> [Split, Convert (retry), Failed]
    workflow.add_conditional_edges(
        "conversion_qa",
        route_after_conversion_qa,
        {
            "split": "split",
            "convert": "convert",
            "failed": "failed",
        }
    )

    # Split -> Split QA
    workflow.add_edge("split", "split_qa")

    # Split QA -> [Clean, Split (retry), Failed]
    workflow.add_conditional_edges(
        "split_qa",
        route_after_split_qa,
        {
            "clean": "clean",
            "split": "split",
            "failed": "failed",
        }
    )

    # Clean -> Chunk -> Generate -> Verify -> Audio QA
    # Each step has conditional routing to handle failures
    workflow.add_conditional_edges(
        "clean",
        route_after_clean,
        {
            "chunk": "chunk",
            "failed": "failed",
        }
    )

    workflow.add_conditional_edges(
        "chunk",
        route_after_chunk,
        {
            "preprocess_tts": "preprocess_tts",
            "failed": "failed",
        }
    )

    workflow.add_conditional_edges(
        "preprocess_tts",
        route_after_preprocess_tts,
        {
            "generate": "generate",
            "failed": "failed",
        }
    )

    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "verify": "verify",
            "failed": "failed",
        }
    )

    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "audio_qa": "audio_qa",
            "failed": "failed",
        }
    )

    # Audio QA -> [Complete, Generate (retry), Failed]
    workflow.add_conditional_edges(
        "audio_qa",
        route_after_audio_qa,
        {
            "complete": "complete",
            "generate": "generate",
            "failed": "failed",
        }
    )

    # Terminal nodes
    workflow.add_edge("complete", END)
    workflow.add_edge("failed", END)

    return workflow.compile()


def run_audiobook_workflow(
    input_file: str,
    output_dir: str,
    voice: str = "Ryan",
    language: str = "English",
    chunk_size: int = 500,
    resume_from: str | None = None,
    workers: int = 1,
) -> AudiobookState:
    """
    Run the complete audiobook generation workflow.

    Args:
        input_file: Path to input document
        output_dir: Directory for output files
        voice: TTS voice to use
        language: Language for TTS
        chunk_size: Maximum characters per TTS chunk
        resume_from: Path to checkpoint file to resume from

    Returns:
        Final workflow state
    """
    # Load or create state
    if resume_from:
        console.print(f"[blue]Resuming from checkpoint: {resume_from}[/blue]")
        state = AudiobookState.load_checkpoint(resume_from)
    else:
        state = AudiobookState(
            input_file=input_file,
            output_dir=output_dir,
            voice=voice,
            language=language,
            chunk_size=chunk_size,
            workers=workers,
        )

    # Build and run the graph
    graph = build_audiobook_graph()

    console.print("\n[bold cyan]Starting Audiobook Generation Workflow[/bold cyan]")
    console.print(f"  Input: {input_file}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Language: {language}")
    console.print(f"  Workers: {state.workers}")

    # Run the workflow
    final_state_dict = graph.invoke(state)

    # Convert back to AudiobookState if needed
    if isinstance(final_state_dict, dict):
        for key, value in final_state_dict.items():
            if hasattr(state, key):
                setattr(state, key, value)
        final_state = state
    else:
        final_state = final_state_dict

    # Save final state
    final_state.save_checkpoint()

    return final_state


def run_chapters_workflow(
    chapter_files: list[str],
    output_dir: str,
    voice: str = "Ryan",
    language: str = "English",
    chunk_size: int = 500,
    resume_from: str | None = None,
    workers: int = 1,
) -> AudiobookState:
    """
    Run audiobook generation from pre-split chapter files.

    This skips the conversion and splitting steps, directly loading
    each file as a chapter.

    Args:
        chapter_files: List of paths to chapter files (in order)
        output_dir: Directory for output files
        voice: TTS voice to use
        language: Language for TTS
        chunk_size: Maximum characters per TTS chunk
        resume_from: Path to checkpoint file to resume from

    Returns:
        Final workflow state
    """
    from pathlib import Path
    from .state import Chapter
    from .nodes.converter import detect_file_type, convert_rtf_to_markdown, convert_html_to_markdown, convert_docx_to_markdown, convert_text_to_markdown

    # Load or create state
    if resume_from:
        console.print(f"[blue]Resuming from checkpoint: {resume_from}[/blue]")
        state = AudiobookState.load_checkpoint(resume_from)
    else:
        state = AudiobookState(
            input_file=chapter_files[0] if chapter_files else "",
            output_dir=output_dir,
            voice=voice,
            language=language,
            chunk_size=chunk_size,
            workers=workers,
        )

        # Load chapters directly from files
        console.print("\n[bold cyan]Loading Chapter Files[/bold cyan]")
        chapters = []

        for i, file_path in enumerate(chapter_files, start=1):
            path = Path(file_path)
            console.print(f"  Loading: {path.name}")

            try:
                file_type = detect_file_type(file_path)

                # Convert based on file type
                if file_type == "rtf":
                    content, metadata = convert_rtf_to_markdown(file_path)
                elif file_type == "html":
                    content, metadata = convert_html_to_markdown(file_path)
                elif file_type == "docx":
                    content, metadata = convert_docx_to_markdown(file_path)
                elif file_type == "markdown":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    metadata = {}
                else:  # plain text
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    content = convert_text_to_markdown(text)
                    metadata = {}

                # Extract chapter title from filename
                # e.g., "accelerando_ch01_lobsters.rtf" -> "Lobsters"
                stem = path.stem
                # Try to extract meaningful title from filename
                parts = stem.replace("_", " ").replace("-", " ").split()
                # Look for chapter name after "ch##" pattern
                title = None
                for j, part in enumerate(parts):
                    if part.lower().startswith("ch") and len(part) <= 4:
                        # Found chapter marker, rest is title
                        title = " ".join(parts[j+1:]).title() if j+1 < len(parts) else f"Chapter {i}"
                        break
                if not title:
                    title = stem.replace("_", " ").replace("-", " ").title()

                chapters.append(Chapter(
                    number=i,
                    title=title,
                    content=content,
                ))
                console.print(f"    ✓ Chapter {i}: {title} ({len(content):,} chars)")

            except Exception as e:
                console.print(f"    [red]✗ Error loading {path.name}: {e}[/red]")
                state.errors.append(f"Failed to load {path.name}: {str(e)}")

        state.chapters = chapters
        state.markdown_content = "\n\n".join(ch.content for ch in chapters)

        # Mark conversion and splitting as passed
        state.conversion_qa_passed = True
        state.split_qa_passed = True
        state.stage = WorkflowStage.CLEANING

    console.print(f"\n[cyan]Loaded {len(state.chapters)} chapters[/cyan]")

    # Build a simplified graph that starts from cleaning
    workflow = StateGraph(AudiobookState)

    # Add nodes (skip convert and split)
    workflow.add_node("clean", node_clean)
    workflow.add_node("chunk", node_chunk)
    workflow.add_node("preprocess_tts", node_preprocess_tts)
    workflow.add_node("generate", node_generate)
    workflow.add_node("verify", node_verify)
    workflow.add_node("audio_qa", node_audio_qa)
    workflow.add_node("complete", node_complete)
    workflow.add_node("failed", node_failed)

    # Set entry point
    workflow.set_entry_point("clean")

    # Add edges with failure handling
    workflow.add_conditional_edges(
        "clean",
        route_after_clean,
        {"chunk": "chunk", "failed": "failed"}
    )
    workflow.add_conditional_edges(
        "chunk",
        route_after_chunk,
        {"preprocess_tts": "preprocess_tts", "failed": "failed"}
    )
    workflow.add_conditional_edges(
        "preprocess_tts",
        route_after_preprocess_tts,
        {"generate": "generate", "failed": "failed"}
    )
    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {"verify": "verify", "failed": "failed"}
    )
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {"audio_qa": "audio_qa", "failed": "failed"}
    )
    workflow.add_conditional_edges(
        "audio_qa",
        route_after_audio_qa,
        {"complete": "complete", "generate": "generate", "failed": "failed"}
    )

    # Terminal nodes
    workflow.add_edge("complete", END)
    workflow.add_edge("failed", END)

    graph = workflow.compile()

    console.print("\n[bold cyan]Starting Audiobook Generation (Pre-split Chapters)[/bold cyan]")
    console.print(f"  Chapters: {len(state.chapters)}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Language: {language}")
    console.print(f"  Workers: {state.workers}")

    # Run the workflow
    final_state_dict = graph.invoke(state)

    # Convert back to AudiobookState if needed
    if isinstance(final_state_dict, dict):
        for key, value in final_state_dict.items():
            if hasattr(state, key):
                setattr(state, key, value)
        final_state = state
    else:
        final_state = final_state_dict

    # Save final state
    final_state.save_checkpoint()

    return final_state


# For direct testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python graph.py <input_file> <output_dir>")
        sys.exit(1)

    result = run_audiobook_workflow(
        input_file=sys.argv[1],
        output_dir=sys.argv[2],
    )

    print(f"\nFinal stage: {result.stage}")
