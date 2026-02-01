"""
Audio verification node - validates generated audio using speech-to-text.
"""

import os
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import soundfile as sf
from rich.console import Console

from ..state import AudiobookState, WorkflowStage

console = Console()

# Global STT model instance
_stt_model = None


def load_stt_model(model_size: str = "base", device: str = "auto"):
    """
    Load the faster-whisper STT model.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v2)
        device: Device to use (auto, cpu, cuda)

    Returns:
        WhisperModel instance
    """
    global _stt_model

    if _stt_model is not None:
        return _stt_model

    console.print(f"[blue]Loading faster-whisper model ({model_size})...[/blue]")

    from faster_whisper import WhisperModel

    # Determine compute type based on device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        elif torch.backends.mps.is_available():
            # MPS not directly supported by faster-whisper, use CPU
            device = "cpu"
            compute_type = "int8"
        else:
            device = "cpu"
            compute_type = "int8"
    elif device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"

    _stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)

    console.print("[green]STT model loaded successfully![/green]")

    return _stt_model


def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Transcribe audio file to text.

    Args:
        audio_path: Path to audio file
        language: Language code

    Returns:
        Transcribed text
    """
    model = load_stt_model()

    segments, info = model.transcribe(
        audio_path,
        language=language[:2].lower(),  # Convert "English" to "en"
        beam_size=5,
        vad_filter=True,
    )

    # Combine all segments
    text = " ".join([segment.text for segment in segments])

    return text.strip()


def transcribe_audio_sample(audio_path: str, start_sec: float = 0, duration_sec: float = 30) -> str:
    """
    Transcribe a sample of the audio file.

    Args:
        audio_path: Path to audio file
        start_sec: Start time in seconds
        duration_sec: Duration to transcribe

    Returns:
        Transcribed text
    """
    # Load audio
    audio, sr = sf.read(audio_path)

    # Extract sample
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    sample = audio[start_sample:min(end_sample, len(audio))]

    # Save temporary file
    temp_path = audio_path + ".sample.wav"
    sf.write(temp_path, sample, sr)

    try:
        text = transcribe_audio(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using sequence matching.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    # Remove punctuation for comparison
    import re
    text1 = re.sub(r"[^\w\s]", "", text1)
    text2 = re.sub(r"[^\w\s]", "", text2)

    # Calculate similarity
    matcher = SequenceMatcher(None, text1.split(), text2.split())
    return matcher.ratio()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate.

    Args:
        reference: Reference text
        hypothesis: Transcribed text

    Returns:
        WER score (lower is better)
    """
    # Normalize
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple Levenshtein distance for words
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def verify_chapter_audio(
    audio_path: str,
    original_text: str,
    sample_duration: float = 30.0,
    similarity_threshold: float = 0.7,
) -> dict:
    """
    Verify audio quality by comparing transcription to original.

    Args:
        audio_path: Path to audio file
        original_text: Original text content
        sample_duration: Duration of sample to verify
        similarity_threshold: Minimum similarity score

    Returns:
        Verification result dict
    """
    if not os.path.exists(audio_path):
        return {
            "passed": False,
            "error": "Audio file not found",
            "similarity": 0.0,
        }

    try:
        # Get audio duration
        info = sf.info(audio_path)
        duration = info.duration

        # Transcribe a sample from the BEGINNING of the audio
        # Middle sampling is unreliable due to variable speaking rates/pauses shifting alignment
        start_time = 0.0
        # Increase duration to 60s for better context
        sample_duration = 60.0
        
        transcribed = transcribe_audio_sample(audio_path, start_time, sample_duration)

        # Get corresponding original text sample
        # We check the first 3000 characters (generous window) to find the match
        original_sample = original_text[:3000]
        
        # Calculate similarity with best substring match
        # instead of direct comparison
        matcher = SequenceMatcher(None, original_sample, transcribed)
        match = matcher.find_longest_match(0, len(original_sample), 0, len(transcribed))
        
        # Use the matched substring for similarity calculation
        if match.size > 10: # If we found a decent match block
            matched_original = original_sample[match.a:match.a + match.size]
            similarity = calculate_similarity(matched_original, transcribed)
        else:
             # Fallback to standard similarity if no clear block found
            similarity = calculate_similarity(original_sample[:len(transcribed)*2], transcribed)

        if 'matched_original' in locals():
             wer = calculate_wer(matched_original, transcribed)
             original_sample = matched_original
        else:
             wer = calculate_wer(original_sample[:len(transcribed)*2], transcribed)
             original_sample = original_sample[:len(transcribed)*2]

        passed = similarity >= similarity_threshold

        return {
            "passed": passed,
            "similarity": similarity,
            "wer": wer,
            "transcribed_sample": transcribed[:200],
            "original_sample": original_sample[:200],
            "duration": duration,
        }

    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "similarity": 0.0,
        }


def verify_audio(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Verify generated audio quality.

    Args:
        state: Current workflow state

    Returns:
        Updated state with verification results
    """
    state.stage = WorkflowStage.VERIFYING

    if not state.chapters:
        state.errors.append("No chapters to verify")
        state.stage = WorkflowStage.FAILED
        return state

    console.print("\n[cyan]Verifying audio quality...[/cyan]")

    verified_count = 0
    failed_chapters = []

    for chapter in state.chapters:
        if not chapter.audio_file:
            continue

        console.print(f"  Verifying Chapter {chapter.number}: {chapter.title[:30]}...")

        # Get original text
        original_text = chapter.cleaned_content or chapter.content

        result = verify_chapter_audio(
            chapter.audio_file,
            original_text,
            sample_duration=30.0,
            similarity_threshold=0.6,  # 60% similarity threshold
        )

        chapter.verified = result["passed"]
        chapter.verification_score = result.get("similarity", 0.0)
        chapter.verification_sample = result.get("transcribed_sample", "")

        if result["passed"]:
            verified_count += 1
            console.print(
                f"    [green]✓ Passed[/green] (similarity: {result['similarity']:.2%})"
            )
        else:
            failed_chapters.append(chapter.number)
            error = result.get("error", f"Low similarity: {result['similarity']:.2%}")
            console.print(f"    [yellow]⚠ Warning[/yellow] ({error})")
            state.warnings.append(
                f"Chapter {chapter.number} verification: {error}"
            )

    # Determine overall pass/fail
    total_with_audio = sum(1 for ch in state.chapters if ch.audio_file)
    pass_rate = verified_count / total_with_audio if total_with_audio > 0 else 0

    if pass_rate >= 0.8:  # 80% of chapters must pass
        state.audio_qa_passed = True
        console.print(f"\n[green]Audio verification passed ({verified_count}/{total_with_audio} chapters)[/green]")
    else:
        state.audio_qa_passed = False
        state.audio_qa_issues.extend([f"Chapter {n} failed verification" for n in failed_chapters])
        console.print(f"\n[yellow]Audio verification: {verified_count}/{total_with_audio} chapters passed[/yellow]")

    state.stage = WorkflowStage.AUDIO_QA

    return state


def get_verification_stats(state: AudiobookState) -> dict:
    """Get verification statistics."""
    verified = [ch for ch in state.chapters if ch.verified]
    with_audio = [ch for ch in state.chapters if ch.audio_file]

    scores = [ch.verification_score for ch in state.chapters if ch.verification_score > 0]

    return {
        "chapters_verified": len(verified),
        "chapters_with_audio": len(with_audio),
        "pass_rate": len(verified) / len(with_audio) if with_audio else 0,
        "avg_similarity": sum(scores) / len(scores) if scores else 0,
        "min_similarity": min(scores) if scores else 0,
        "max_similarity": max(scores) if scores else 0,
    }
