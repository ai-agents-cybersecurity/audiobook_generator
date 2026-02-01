"""
CLI interface for the audiobook generator.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .graph import run_audiobook_workflow, run_chapters_workflow
from .state import AudiobookState, WorkflowStage
from .config import config
import glob as glob_module

app = typer.Typer(
    name="audiobook-generator",
    help="Convert books to audiobooks using Qwen3-TTS",
    add_completion=False,
)

console = Console()

# Available voices for CustomVoice model
AVAILABLE_VOICES = [
    "Ryan",    # English male - best for audiobooks
    "Aiden",   # English male
    "Vivian",  # Chinese/English female
    "Serena",  # Chinese/English female
]


def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           🎧 Audiobook Generator                          ║
    ║        Powered by Qwen3-TTS & LangGraph                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print(f"    [dim]QA Model: {config.OLLAMA_MODEL}[/dim]")
    console.print()


@app.command()
def generate(
    input_file: str = typer.Argument(
        ...,
        help="Input file path (RTF, TXT, MD, HTML, or DOCX)",
    ),
    output_dir: str = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: ./audiobook_output)",
    ),
    voice: str = typer.Option(
        "Ryan",
        "--voice", "-v",
        help=f"TTS voice to use. Options: {', '.join(AVAILABLE_VOICES)}",
    ),
    language: str = typer.Option(
        "English",
        "--language", "-l",
        help="Language for TTS (English, Chinese, etc.)",
    ),
    chunk_size: int = typer.Option(
        500,
        "--chunk-size", "-c",
        help="Maximum characters per TTS chunk (300-800 recommended)",
    ),
    resume: str = typer.Option(
        None,
        "--resume", "-r",
        help="Resume from checkpoint file",
    ),
    skip_qa: bool = typer.Option(
        False,
        "--skip-qa",
        help="Skip QA verification steps (faster but less reliable)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run conversion and splitting only, no audio generation",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of parallel workers for TTS generation",
    ),
):
    """
    Generate an audiobook from a text document.

    Examples:
        audiobook-generator generate mybook.rtf
        audiobook-generator generate mybook.txt -o ./my_audiobook -v Aiden
        audiobook-generator generate mybook.md --resume checkpoint.json
    """
    print_banner()

    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Set default output directory
    if output_dir is None:
        output_dir = f"./audiobook_output_{input_path.stem}"

    # Validate voice
    if voice not in AVAILABLE_VOICES:
        console.print(f"[yellow]Warning: Unknown voice '{voice}'. Using 'Ryan'.[/yellow]")
        voice = "Ryan"

    # Validate chunk size
    if chunk_size < 100 or chunk_size > 1000:
        console.print(f"[yellow]Warning: Chunk size {chunk_size} is outside recommended range. Using 500.[/yellow]")
        chunk_size = 500

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Show configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input File", str(input_path.absolute()))
    config_table.add_row("Output Directory", str(Path(output_dir).absolute()))
    config_table.add_row("Voice", voice)
    config_table.add_row("Language", language)
    config_table.add_row("Chunk Size", str(chunk_size))
    config_table.add_row("Workers", str(workers))
    config_table.add_row("Resume From", resume or "New generation")
    console.print(config_table)
    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN: Will only convert and split, no audio generation[/yellow]\n")

    try:
        # Run the workflow
        final_state = run_audiobook_workflow(
            input_file=str(input_path.absolute()),
            output_dir=str(Path(output_dir).absolute()),
            voice=voice,
            language=language,
            chunk_size=chunk_size,
            resume_from=resume,
            workers=workers,
        )

        # Print results
        print_results(final_state)

        if final_state.stage == WorkflowStage.COMPLETE:
            console.print("\n[bold green]✓ Audiobook generation completed successfully![/bold green]")
            raise typer.Exit(0)
        else:
            console.print("\n[bold red]✗ Audiobook generation failed[/bold red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted. Checkpoint saved.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="generate-chapters")
def generate_chapters(
    chapter_dir: str = typer.Argument(
        ...,
        help="Directory containing chapter files (e.g., chapter_01.rtf, chapter_02.rtf)",
    ),
    output_dir: str = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: ./audiobook_output)",
    ),
    pattern: str = typer.Option(
        "*.rtf",
        "--pattern", "-p",
        help="Glob pattern for chapter files (default: *.rtf)",
    ),
    voice: str = typer.Option(
        "Ryan",
        "--voice", "-v",
        help=f"TTS voice to use. Options: {', '.join(AVAILABLE_VOICES)}",
    ),
    language: str = typer.Option(
        "English",
        "--language", "-l",
        help="Language for TTS (English, Chinese, etc.)",
    ),
    chunk_size: int = typer.Option(
        500,
        "--chunk-size", "-c",
        help="Maximum characters per TTS chunk (300-800 recommended)",
    ),
    resume: str = typer.Option(
        None,
        "--resume", "-r",
        help="Resume from checkpoint file",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of parallel workers for TTS generation",
    ),
):
    """
    Generate audiobook from pre-split chapter files.

    This mode skips automatic chapter detection and uses files directly.
    Files are processed in alphabetical order, so name them accordingly
    (e.g., ch01_intro.rtf, ch02_beginning.rtf, etc.)

    Examples:
        audiobook-generator generate-chapters ./mybook/
        audiobook-generator generate-chapters ./chapters/ -p "chapter_*.txt"
        audiobook-generator generate-chapters ./book/ --voice Aiden
    """
    print_banner()

    # Validate directory
    chapter_path = Path(chapter_dir)
    if not chapter_path.exists():
        console.print(f"[red]Error: Directory not found: {chapter_dir}[/red]")
        raise typer.Exit(1)

    # Find chapter files
    if chapter_path.is_file():
        # Single file provided - treat as pattern
        chapter_files = [chapter_path]
    else:
        # Directory - find files matching pattern
        search_pattern = str(chapter_path / pattern)
        chapter_files = sorted(glob_module.glob(search_pattern))

    if not chapter_files:
        console.print(f"[red]Error: No files found matching pattern '{pattern}' in {chapter_dir}[/red]")
        raise typer.Exit(1)

    # Set default output directory
    if output_dir is None:
        output_dir = f"./audiobook_output_{chapter_path.stem}"

    # Validate voice
    if voice not in AVAILABLE_VOICES:
        console.print(f"[yellow]Warning: Unknown voice '{voice}'. Using 'Ryan'.[/yellow]")
        voice = "Ryan"

    # Validate chunk size
    if chunk_size < 100 or chunk_size > 1000:
        console.print(f"[yellow]Warning: Chunk size {chunk_size} is outside recommended range. Using 500.[/yellow]")
        chunk_size = 500

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Show configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Chapter Directory", str(chapter_path.absolute()))
    config_table.add_row("Chapter Files", str(len(chapter_files)))
    config_table.add_row("Output Directory", str(Path(output_dir).absolute()))
    config_table.add_row("Voice", voice)
    config_table.add_row("Language", language)
    config_table.add_row("Chunk Size", str(chunk_size))
    config_table.add_row("Workers", str(workers))
    console.print(config_table)
    console.print()

    # Show found files
    console.print("[cyan]Chapter files found:[/cyan]")
    for i, f in enumerate(chapter_files, 1):
        console.print(f"  {i}. {Path(f).name}")
    console.print()

    try:
        # Run the chapters workflow
        final_state = run_chapters_workflow(
            chapter_files=[str(f) for f in chapter_files],
            output_dir=str(Path(output_dir).absolute()),
            voice=voice,
            language=language,
            chunk_size=chunk_size,
            resume_from=resume,
            workers=workers,
        )

        # Print results
        print_results(final_state)

        if final_state.stage == WorkflowStage.COMPLETE:
            console.print("\n[bold green]✓ Audiobook generation completed successfully![/bold green]")
            raise typer.Exit(0)
        else:
            console.print("\n[bold red]✗ Audiobook generation failed[/bold red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted. Checkpoint saved.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        import traceback
        console.print(f"\n[red]Error: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def info(
    input_file: str = typer.Argument(
        ...,
        help="Input file to analyze",
    ),
):
    """
    Analyze an input file without generating audio.

    Shows document structure, detected chapters, and estimated output.
    """
    print_banner()

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    from .nodes.converter import convert_document, get_conversion_stats
    from .nodes.splitter import split_chapters, get_split_stats

    # Create temporary state
    state = AudiobookState(
        input_file=str(input_path.absolute()),
        output_dir="/tmp/audiobook_info",
    )

    # Convert
    console.print("[cyan]Analyzing document...[/cyan]")
    state = convert_document(state)

    if state.stage == WorkflowStage.FAILED:
        console.print(f"[red]Conversion failed: {state.errors}[/red]")
        raise typer.Exit(1)

    # Get conversion stats
    conv_stats = get_conversion_stats(state)

    # Split
    state = split_chapters(state)

    # Get split stats
    split_stats = get_split_stats(state)

    # Display results
    console.print()

    # Document info
    doc_table = Table(title="Document Information", show_header=False)
    doc_table.add_column("Property", style="cyan")
    doc_table.add_column("Value", style="green")
    doc_table.add_row("Title", state.book_title or "Not detected")
    doc_table.add_row("Author", state.book_author or "Not detected")
    doc_table.add_row("Total Characters", f"{conv_stats['total_characters']:,}")
    doc_table.add_row("Total Words", f"{conv_stats['total_words']:,}")
    doc_table.add_row("Paragraphs", str(conv_stats['total_paragraphs']))
    console.print(doc_table)
    console.print()

    # Chapter info
    chapter_table = Table(title="Chapter Structure")
    chapter_table.add_column("#", style="cyan")
    chapter_table.add_column("Title", style="green")
    chapter_table.add_column("Characters", justify="right")
    chapter_table.add_column("Est. Duration", justify="right")

    for ch in state.chapters[:20]:
        # Estimate duration: ~150 words/minute, ~5 chars/word
        est_words = len(ch.content) / 5
        est_minutes = est_words / 150
        chapter_table.add_row(
            str(ch.number),
            ch.title[:40],
            f"{len(ch.content):,}",
            f"{est_minutes:.1f} min",
        )

    if len(state.chapters) > 20:
        chapter_table.add_row("...", f"({len(state.chapters) - 20} more)", "", "")

    console.print(chapter_table)
    console.print()

    # Estimates
    total_chars = sum(len(ch.content) for ch in state.chapters)
    total_words = total_chars / 5
    total_minutes = total_words / 150

    est_table = Table(title="Estimates", show_header=False)
    est_table.add_column("Property", style="cyan")
    est_table.add_column("Value", style="green")
    est_table.add_row("Total Chapters", str(len(state.chapters)))
    est_table.add_row("Estimated Duration", f"{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    est_table.add_row("Estimated Chunks", str(total_chars // 500))
    console.print(est_table)


@app.command()
def resume(
    checkpoint_file: str = typer.Argument(
        ...,
        help="Checkpoint file to resume from",
    ),
):
    """
    Resume a previous audiobook generation from checkpoint.
    """
    print_banner()

    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        console.print(f"[red]Error: Checkpoint file not found: {checkpoint_file}[/red]")
        raise typer.Exit(1)

    try:
        state = AudiobookState.load_checkpoint(str(checkpoint_path))

        console.print(f"[green]Loaded checkpoint from: {checkpoint_file}[/green]")
        console.print(f"  Stage: {state.stage.value}")
        console.print(f"  Chapters: {len(state.chapters)}")
        console.print(f"  Progress: {state.processed_chunks}/{state.total_chunks} chunks")

        # Continue the workflow
        final_state = run_audiobook_workflow(
            input_file=state.input_file,
            output_dir=state.output_dir,
            voice=state.voice,
            language=state.language,
            resume_from=str(checkpoint_path),
        )

        print_results(final_state)

    except Exception as e:
        console.print(f"[red]Error resuming: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def voices():
    """
    List available TTS voices.
    """
    print_banner()

    voice_table = Table(title="Available Voices")
    voice_table.add_column("Voice", style="cyan")
    voice_table.add_column("Gender", style="green")
    voice_table.add_column("Languages", style="yellow")
    voice_table.add_column("Best For", style="white")

    voice_table.add_row("Ryan", "Male", "English", "Audiobooks, narration (recommended)")
    voice_table.add_row("Aiden", "Male", "English", "Audiobooks, casual reading")
    voice_table.add_row("Vivian", "Female", "Chinese, English", "Multilingual content")
    voice_table.add_row("Serena", "Female", "Chinese, English", "Multilingual content")

    console.print(voice_table)


@app.command(name="show-config")
def show_config():
    """
    Show current configuration from .env.audiobook file.
    """
    print_banner()

    config_table = Table(title="Current Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Source", style="dim")

    config_table.add_row("OLLAMA_MODEL", config.OLLAMA_MODEL, ".env.audiobook")
    config_table.add_row("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL, ".env.audiobook")
    config_table.add_row("TTS_VOICE", config.TTS_VOICE, ".env.audiobook")
    config_table.add_row("TTS_LANGUAGE", config.TTS_LANGUAGE, ".env.audiobook")
    config_table.add_row("TTS_CHUNK_SIZE", str(config.TTS_CHUNK_SIZE), ".env.audiobook")
    config_table.add_row("MAX_QA_ATTEMPTS", str(config.MAX_QA_ATTEMPTS), ".env.audiobook")

    console.print(config_table)

    env_file = Path(__file__).parent / ".env.audiobook"
    console.print(f"\n[dim]Config file: {env_file}[/dim]")


def print_results(state: AudiobookState):
    """Print workflow results."""
    console.print()

    # Results table
    results_table = Table(title="Generation Results")
    results_table.add_column("Chapter", style="cyan")
    results_table.add_column("Title", style="green")
    results_table.add_column("Status", style="white")
    results_table.add_column("Verified", style="yellow")

    for ch in state.chapters:
        status = "✓" if ch.audio_file else "✗"
        verified = f"{ch.verification_score:.0%}" if ch.verification_score > 0 else "-"
        results_table.add_row(
            str(ch.number),
            ch.title[:30],
            status,
            verified,
        )

    console.print(results_table)

    # Warnings
    if state.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in state.warnings[:10]:
            console.print(f"  - {warning}")
        if len(state.warnings) > 10:
            console.print(f"  ... and {len(state.warnings) - 10} more")

    # Errors
    if state.errors:
        console.print("\n[red]Errors:[/red]")
        for error in state.errors:
            console.print(f"  - {error}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
