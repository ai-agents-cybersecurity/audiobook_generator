"""
Configuration management for Audiobook Generator.
Loads settings from .env.audiobook file.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load .env.audiobook from the package directory
_env_file = Path(__file__).parent / ".env.audiobook"
if _env_file.exists():
    load_dotenv(_env_file)


def strip_thinking_tags(content: str) -> str:
    """
    Strip <think>...</think> tags from thinking model output.
    Returns the content after the thinking block.

    Args:
        content: Raw model output that may contain thinking tags

    Returns:
        Content with thinking tags removed
    """
    if not content:
        return content

    # Remove <think>...</think> blocks (handles multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # Also handle unclosed <think> tags (model still thinking)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)

    return cleaned.strip()


def extract_json_from_response(content: str) -> Optional[dict]:
    """
    Extract JSON object from model response, handling thinking tags.

    Args:
        content: Raw model output

    Returns:
        Parsed JSON dict or None if not found
    """
    import json

    # First strip thinking tags
    cleaned = strip_thinking_tags(content)

    # Try to find JSON object (handles nested braces better)
    # Look for outermost { }
    brace_count = 0
    start_idx = None

    for i, char in enumerate(cleaned):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = cleaned[start_idx:i+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

    # Fallback: simple regex
    json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


class Config:
    """Configuration settings loaded from environment."""

    # Ollama settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

    # TTS settings
    TTS_VOICE: str = os.getenv("TTS_VOICE", "Ryan")
    TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "English")
    TTS_CHUNK_SIZE: int = int(os.getenv("TTS_CHUNK_SIZE", "500"))
    TTS_WORKERS: int = int(os.getenv("TTS_WORKERS", "1"))
    TTS_PREPROCESSING_ENABLED: bool = os.getenv("TTS_PREPROCESSING_ENABLED", "true").lower() == "true"
    QWEN3_TTS_TAGS_PATH: str = os.getenv("QWEN3_TTS_TAGS_PATH", str(Path(__file__).parent / "qwen3_tts_tags.json"))  # Legacy
    QWEN3_TTS_CONFIG_PATH: str = os.getenv("QWEN3_TTS_CONFIG_PATH", str(Path(__file__).parent / "qwen3_tts_config.json"))

    # QA settings
    MAX_QA_ATTEMPTS: int = int(os.getenv("MAX_QA_ATTEMPTS", "3"))

    # Optimization settings
    # Default to True on Mac, False otherwise
    USE_MLX: bool = os.getenv("USE_MLX", "true").lower() == "true" if sys.platform == "darwin" and os.uname().machine == "arm64" else False

    @classmethod
    def get_ollama_model(cls) -> str:
        """Get the configured Ollama model name."""
        return cls.OLLAMA_MODEL

    @classmethod
    def get_ollama_base_url(cls) -> str:
        """Get the Ollama base URL."""
        return cls.OLLAMA_BASE_URL

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("Audiobook Generator Configuration:")
        print(f"  OLLAMA_MODEL: {cls.OLLAMA_MODEL}")
        print(f"  OLLAMA_BASE_URL: {cls.OLLAMA_BASE_URL}")
        print(f"  TTS_VOICE: {cls.TTS_VOICE}")
        print(f"  TTS_LANGUAGE: {cls.TTS_LANGUAGE}")
        print(f"  TTS_CHUNK_SIZE: {cls.TTS_CHUNK_SIZE}")
        print(f"  TTS_WORKERS: {cls.TTS_WORKERS}")
        print(f"  TTS_PREPROCESSING_ENABLED: {cls.TTS_PREPROCESSING_ENABLED}")
        print(f"  QWEN3_TTS_TAGS_PATH: {cls.QWEN3_TTS_TAGS_PATH}")
        print(f"  MAX_QA_ATTEMPTS: {cls.MAX_QA_ATTEMPTS}")


# Singleton instance
config = Config()
