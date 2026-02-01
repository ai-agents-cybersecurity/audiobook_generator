"""
TTS generation node using MLX-Audio (optimized for Apple Silicon).
"""

from typing import Tuple
import numpy as np
import logging
import warnings
from rich.console import Console

# Import mlx-audio
try:
    from mlx_audio.tts import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

console = Console()

_model = None
_tokenizer = None
_config = None

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

def is_mlx_available() -> bool:
    return MLX_AVAILABLE

def load_mlx_model():
    """Load the MLX Qwen3-TTS model."""
    global _model, _tokenizer, _config
    
    if _model is not None:
        return _model, _tokenizer, _config

    if not MLX_AVAILABLE:
        raise ImportError("mlx-audio is not installed. Please install it with `pip install mlx-audio`.")

    console.print(f"[blue]Loading MLX Qwen3-TTS model: {MODEL_ID}...[/blue]")
    
    try:
        # Suppress transformers warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, message=".*incorrect regex pattern.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*model of type qwen3_tts.*")
        
        # Load model using mlx_audio API

        # The Model class inherits from dict but IS the model itself
        loaded = load_model(MODEL_ID)

        # The loaded object IS the model (it inherits from dict but is the Model class)
        # Access tokenizer and config as attributes, not dict keys
        _model = loaded
        _tokenizer = getattr(loaded, 'tokenizer', None)
        _config = getattr(loaded, 'config', None)

        console.print("[green]MLX TTS model loaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error loading MLX TTS model: {e}[/red]")
        raise e

    return _model, _tokenizer, _config

def generate_chunk_audio_mlx(
    text: str,
    speaker: str = "Ryan",
    language: str = "English",
    instruct: str = "Read in a clear, engaging audiobook narration style.",
) -> Tuple[np.ndarray, int]:
    """
    Generate audio for a text chunk using MLX.
    
    Args:
        text: Text to synthesize
        speaker: Speaker name
        language: Language (e.g., "English", "Chinese")
        instruct: Natural language instruction for speech style
                   
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    model, tokenizer, config = load_mlx_model()

    if model is None:
        raise RuntimeError("Failed to load model (model is None)")

    # Check if speaker is valid (model returns lowercase speakers)
    available_speakers = model.get_supported_speakers() if hasattr(model, 'get_supported_speakers') else []
    speaker_lower = speaker.lower()
    if available_speakers and speaker_lower not in available_speakers:
        console.print(f"[yellow]Warning: Speaker '{speaker}' not found in {available_speakers}. Using first available.[/yellow]")
        speaker_lower = available_speakers[0] if available_speakers else speaker_lower
    
    console.print(f"[dim]Generating with MLX ({speaker_lower})...[/dim]")
    
    # Generate speech
    # The Qwen3 model in mlx-audio has generate_custom_voice
    
    # generate_custom_voice returns a generator of GenerationResult
    # We need to collect the audio
    
    full_audio = []
    sr = 24000 # default for Qwen3, but will check result

    try:
        # returns generator
        gen = model.generate_custom_voice(
            text=text,
            speaker=speaker_lower,
            language=language.lower(),
            instruct=instruct,
            verbose=False
        )
        
        for result in gen:
            # result is GenerationResult(audio=..., sample_rate=..., ...)
            if result.audio is not None:
                # audio is MLX array - convert to numpy
                audio_data = result.audio
                if hasattr(audio_data, 'tolist'):
                    # MLX array - convert via tolist() for safety
                    chunk = np.array(audio_data.tolist(), dtype=np.float32)
                elif isinstance(audio_data, np.ndarray):
                    chunk = audio_data.astype(np.float32)
                else:
                    chunk = np.array(audio_data, dtype=np.float32)

                full_audio.append(chunk)
                sr = result.sample_rate
                
    except Exception as e:
        console.print(f"[red]Error during MLX generation: {e}[/red]")
        # Fallback or re-raise
        raise e
        
    if not full_audio:
        return np.array([], dtype=np.float32), sr
        
    audio = np.concatenate(full_audio)
    
    if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, sr
