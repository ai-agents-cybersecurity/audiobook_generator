"""
TTS generation node - generates audio using Qwen3-TTS (MLX optimized for Apple Silicon).
"""

import os
from typing import Optional

import numpy as np
import soundfile as sf
from rich.console import Console

from ..state import AudiobookState, WorkflowStage, TTSChunk
from .tts_mlx import load_mlx_model, is_mlx_available, generate_chunk_audio_mlx

console = Console()

# Global model loading is handled in tts_mlx


def _chapter_tts_chunks(chapter) -> list[TTSChunk]:
    """Get TTS chunks, falling back to plain text chunks with default instruct."""
    if chapter.tts_chunks:
        return chapter.tts_chunks
    # Fallback: wrap plain text chunks in TTSChunk with default instruct
    default_instruct = "Read in a clear, engaging audiobook narration style."
    return [TTSChunk(text=chunk, instruct=default_instruct) for chunk in chapter.chunks]

def generate_chapter_audio(
    state: AudiobookState,
    chapter_index: int,
    output_dir: str,
    progress_callback: Optional[callable] = None,
) -> tuple[str, bool]:
    """
    Generate audio for a single chapter.

    Args:
        state: Workflow state
        chapter_index: Index of chapter to generate
        output_dir: Directory to save audio
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (output file path, success)
    """
    chapter = state.chapters[chapter_index]
    tts_chunks = _chapter_tts_chunks(chapter)

    if not tts_chunks:
        console.print(f"[yellow]Chapter {chapter.number} has no chunks, skipping[/yellow]")
        return None, False

    console.print(f"[blue]Chapter {chapter.number}: {len(tts_chunks)} chunks to process[/blue]")

    if not is_mlx_available():
        error_msg = "MLX is not available. Please install mlx-audio."
        console.print(f"[red]{error_msg}[/red]")
        state.errors.append(error_msg)
        return None, False

    # Ensure model is loaded
    try:
        load_mlx_model()
    except Exception as e:
        console.print(f"[red]Failed to load MLX model: {e}[/red]")
        state.errors.append(f"MLX model load failed: {str(e)}")
        return None, False

    # Generate audio for each chunk
    audio_chunks = []
    sr = None

    for i, tts_chunk in enumerate(tts_chunks):
        if not tts_chunk.text.strip():
            continue

        try:
            audio, sample_rate = generate_chunk_audio_mlx(
                text=tts_chunk.text,
                speaker=state.voice,
                language=state.language,
                instruct=tts_chunk.instruct,
            )
            
            audio_chunks.append(audio)

            if sr is None:
                sr = sample_rate

            state.processed_chunks += 1

            if progress_callback:
                progress_callback(i + 1, len(tts_chunks), tts_chunk.text[:50])

        except Exception as e:
            console.print(f"[red]Chunk {i} error: {str(e)}[/red]")
            state.warnings.append(f"Chapter {chapter.number}, chunk {i}: {str(e)}")
            # Print first error with traceback for debugging
            if i == 0:
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
            continue
    
    if not audio_chunks or sr is None:
        console.print(f"[red]No audio generated for chapter {chapter.number}[/red]")
        return None, False

    # Concatenate chunks
    full_audio = concatenate_audio_chunks(audio_chunks, sr)

    # Normalize
    full_audio = normalize_audio(full_audio)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in chapter.title)
    safe_title = safe_title[:50].strip()
    filename = f"chapter_{chapter.number:03d}_{safe_title}.wav"
    output_path = os.path.join(output_dir, filename)

    sf.write(output_path, full_audio, sr)

    chapter.audio_file = output_path
    state.sample_rate = sr

    return output_path, True


def concatenate_audio_chunks(chunks: list[np.ndarray], sr: int, silence_ms: int = 300) -> np.ndarray:
    """Concatenate audio chunks with silence between them."""
    if not chunks:
        return np.array([], dtype=np.float32)

    # Create silence buffer
    silence_samples = int(sr * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    # Concatenate with silence
    result_parts = []
    for i, chunk in enumerate(chunks):
        result_parts.append(chunk)
        if i < len(chunks) - 1:
            result_parts.append(silence)

    return np.concatenate(result_parts)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    if len(audio) == 0:
        return audio

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)

    # Scale audio
    scale = target_rms / rms
    return audio * scale


def generate_audio(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Generate audio for all chapters.
    """
    state.stage = WorkflowStage.GENERATING

    if not state.chapters:
        state.errors.append("No chapters to generate audio for")
        state.stage = WorkflowStage.FAILED
        return state

    # Ensure output directory exists
    output_dir = os.path.join(state.output_dir, "audio")
    os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[cyan]Total chapters: {len(state.chapters)}, Total chunks: {state.total_chunks}[/cyan]")

    try:
        # Pre-load the TTS model in the main process to check availability
        console.print("[blue]Pre-loading MLX TTS model...[/blue]")
        if is_mlx_available():
            load_mlx_model()
            console.print("[green]MLX TTS model ready![/green]")
        else:
            console.print("[red]MLX not available! Cannot proceed.[/red]")
            state.errors.append("MLX not available")
            state.stage = WorkflowStage.FAILED
            return state

        # Prepare arguments for parallel execution
        chapter_args = []
        for i, chapter in enumerate(state.chapters):
            if not _chapter_tts_chunks(chapter):
                continue
            chapter_args.append({
                "chapter_index": i,
                "chapter_title": chapter.title,
                "chunks": _chapter_tts_chunks(chapter),
                "voice": state.voice,
                "language": state.language,
                "output_dir": os.path.join(state.output_dir, "audio"),
                "chapter_number": chapter.number
            })

        results = []
        if state.workers > 1 and len(chapter_args) > 1:
            console.print(f"\n[bold green]Running parallel generation with {state.workers} workers...[/bold green]")
            import concurrent.futures
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=state.workers) as executor:
                # Submit all tasks
                future_to_chapter = {
                    executor.submit(generate_chapter_worker, args): args["chapter_index"] 
                    for args in chapter_args
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_chapter):
                    idx = future_to_chapter[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update state
                        output_path, success, error = result
                        chapter = state.chapters[idx]
                        
                        if success:
                            chapter.audio_file = output_path
                            console.print(f"  [green]✓ {chapter.title[:30]}: Saved to {os.path.basename(output_path)}[/green]")
                        else:
                            console.print(f"  [red]✗ {chapter.title[:30]}: Failed - {error}[/red]")
                            state.warnings.append(f"Failed to generate audio for chapter {chapter.number}: {error}")
                            
                        # Save checkpoint periodically
                        state.save_checkpoint()
                        
                    except Exception as e:
                        console.print(f"[red]Worker error for chapter index {idx}: {e}[/red]")
                        state.errors.append(f"Worker error: {str(e)}")
        else:
            # Sequential execution for single worker
            console.print("\n[blue]Running sequential generation...[/blue]")
            for i, chapter in enumerate(state.chapters):
                state.current_chapter_index = i
                tts_chunks = _chapter_tts_chunks(chapter)
                console.print(f"\n[cyan]Processing Chapter {chapter.number}: {chapter.title[:40]}...[/cyan]")
                console.print(f"  Chunks: {len(tts_chunks)}")

                if not tts_chunks:
                    console.print(f"  [yellow]No chunks, skipping[/yellow]")
                    continue

                output_path, success = generate_chapter_audio(
                    state, i, output_dir, None
                )

                if success:
                    console.print(f"  [green]✓ Saved to {output_path}[/green]")
                else:
                    console.print(f"  [red]✗ Failed to generate audio[/red]")
                    state.warnings.append(f"Failed to generate audio for chapter {chapter.number}")

                # Save checkpoint after each chapter
                state.save_checkpoint()

        state.stage = WorkflowStage.VERIFYING

    except Exception as e:
        import traceback
        console.print(f"[red]Audio generation error: {str(e)}[/red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
        state.errors.append(f"Audio generation error: {str(e)}")
        state.stage = WorkflowStage.FAILED

    return state


def generate_chapter_worker(args: dict) -> tuple[Optional[str], bool, Optional[str]]:
    """
    Worker function for parallel chapter generation.
    Must be picklable and self-contained.
    """
    try:
        # Import here to avoid issues with pickling in the main process if imported at top level
        # though top-level imports are usually fine if modules are importable
        from .tts_mlx import load_mlx_model, generate_chunk_audio_mlx
        import numpy as np
        import soundfile as sf
        import os
        
        # Load model in this process
        # MLX handles lazy loading, but we need to ensure it's ready
        load_mlx_model()
        
        chunks = args["chunks"]
        voice = args["voice"]
        language = args["language"]
        output_dir = args["output_dir"]
        chapter_title = args["chapter_title"]
        chapter_number = args["chapter_number"]
        
        # Generate audio for chunks
        audio_chunks = []
        sr = None
        
        # Create a local console for the worker
        console = Console()
        
        for i, tts_chunk in enumerate(chunks):
            # Print progress every 10 chunks or for the first few
            if i % 10 == 0 or i < 5:
                # Calculate percent
                pct = (i / len(chunks)) * 100
                console.print(f"[dim]Chapter {chapter_number}: Processing chunk {i+1}/{len(chunks)} ({pct:.1f}%)[/dim]")

            # Handle both TTSChunk objects and legacy string chunks
            if hasattr(tts_chunk, 'text'):
                chunk_text = tts_chunk.text
                chunk_instruct = tts_chunk.instruct
            else:
                # Legacy: plain string
                chunk_text = tts_chunk
                chunk_instruct = "Read in a clear, engaging audiobook narration style."

            if not chunk_text.strip():
                continue
                
            audio, sample_rate = generate_chunk_audio_mlx(
                text=chunk_text,
                speaker=voice,
                language=language,
                instruct=chunk_instruct,
            )
            
            audio_chunks.append(audio)
            if sr is None:
                sr = sample_rate
                
        if not audio_chunks or sr is None:
            return None, False, "No audio generated"
            
        # Concatenate and normalize (local simplified versions if needed, or import)
        full_audio = concatenate_audio_chunks(audio_chunks, sr)
        full_audio = normalize_audio(full_audio)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in chapter_title)
        safe_title = safe_title[:50].strip()
        filename = f"chapter_{chapter_number:03d}_{safe_title}.wav"
        output_path = os.path.join(output_dir, filename)
        
        sf.write(output_path, full_audio, sr)
        
        return output_path, True, None
        
    except Exception as e:
        return None, False, str(e)



def get_generation_stats(state: AudiobookState) -> dict:
    """Get statistics about audio generation."""
    generated = [ch for ch in state.chapters if ch.audio_file]

    return {
        "chapters_generated": len(generated),
        "chapters_total": len(state.chapters),
        "chunks_processed": state.processed_chunks,
        "chunks_total": state.total_chunks,
        "audio_files": [ch.audio_file for ch in generated if ch.audio_file],
    }
