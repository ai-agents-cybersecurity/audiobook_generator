"""
State definitions for the LangGraph audiobook workflow.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import json


class WorkflowStage(str, Enum):
    """Stages in the audiobook generation workflow."""
    INIT = "init"
    CONVERTING = "converting"
    CONVERSION_QA = "conversion_qa"
    SPLITTING = "splitting"
    SPLIT_QA = "split_qa"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    GENERATING = "generating"
    VERIFYING = "verifying"
    AUDIO_QA = "audio_qa"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Chapter:
    """Represents a single chapter."""
    number: int
    title: str
    content: str
    cleaned_content: Optional[str] = None
    chunks: list[str] = field(default_factory=list)
    audio_file: Optional[str] = None
    verified: bool = False
    verification_score: float = 0.0
    verification_sample: Optional[str] = None

    def to_dict(self, include_content: bool = True) -> dict:
        """Serialize chapter to dict. Set include_content=False for summary only."""
        d = {
            "number": self.number,
            "title": self.title,
            "audio_file": self.audio_file,
            "verified": self.verified,
            "verification_score": self.verification_score,
        }
        if include_content:
            d["content"] = self.content
            d["cleaned_content"] = self.cleaned_content
            d["chunks"] = self.chunks
        else:
            d["content_length"] = len(self.content) if self.content else 0
            d["cleaned_content_length"] = len(self.cleaned_content) if self.cleaned_content else 0
            d["num_chunks"] = len(self.chunks)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chapter":
        """Restore chapter from dict."""
        return cls(
            number=d["number"],
            title=d["title"],
            content=d.get("content", ""),
            cleaned_content=d.get("cleaned_content"),
            chunks=d.get("chunks", []),
            audio_file=d.get("audio_file"),
            verified=d.get("verified", False),
            verification_score=d.get("verification_score", 0.0),
            verification_sample=d.get("verification_sample"),
        )


@dataclass
class AudiobookState:
    """
    Complete state for the audiobook generation workflow.
    This state is passed through all nodes in the LangGraph workflow.
    """
    # Input configuration
    input_file: str
    output_dir: str
    voice: str = "Ryan"
    language: str = "English"

    # Current workflow stage
    stage: WorkflowStage = WorkflowStage.INIT

    # Document content at various stages
    raw_content: Optional[str] = None
    markdown_content: Optional[str] = None

    # Chapters
    chapters: list[Chapter] = field(default_factory=list)
    current_chapter_index: int = 0

    # QA state
    conversion_qa_passed: bool = False
    conversion_qa_issues: list[str] = field(default_factory=list)
    conversion_qa_attempts: int = 0

    split_qa_passed: bool = False
    split_qa_issues: list[str] = field(default_factory=list)
    split_qa_attempts: int = 0

    audio_qa_passed: bool = False
    audio_qa_issues: list[str] = field(default_factory=list)
    audio_qa_attempts: int = 0

    # Generation progress
    total_chunks: int = 0
    processed_chunks: int = 0

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metadata
    book_title: Optional[str] = None
    book_author: Optional[str] = None

    # Configuration
    max_qa_attempts: int = 3
    chunk_size: int = 500  # characters per chunk for TTS
    sample_rate: int = 24000
    workers: int = 1

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> str:
        """Save current state to a checkpoint file for resumption."""
        if checkpoint_path is None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(Path(self.output_dir) / "checkpoint.json")

        state_dict = {
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "voice": self.voice,
            "language": self.language,
            "stage": self.stage.value,
            "markdown_content": self.markdown_content,
            "chapters": [ch.to_dict(include_content=True) for ch in self.chapters],
            "current_chapter_index": self.current_chapter_index,
            "conversion_qa_passed": self.conversion_qa_passed,
            "conversion_qa_attempts": self.conversion_qa_attempts,
            "split_qa_passed": self.split_qa_passed,
            "split_qa_attempts": self.split_qa_attempts,
            "audio_qa_passed": self.audio_qa_passed,
            "audio_qa_attempts": self.audio_qa_attempts,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "errors": self.errors,
            "warnings": self.warnings,
            "book_title": self.book_title,
            "book_author": self.book_author,
            "chunk_size": self.chunk_size,
            "workers": self.workers,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "AudiobookState":
        """Load state from a checkpoint file."""
        with open(checkpoint_path, "r") as f:
            state_dict = json.load(f)

        state = cls(
            input_file=state_dict["input_file"],
            output_dir=state_dict["output_dir"],
            voice=state_dict.get("voice", "Ryan"),
            language=state_dict.get("language", "English"),
        )
        state.stage = WorkflowStage(state_dict["stage"])
        state.markdown_content = state_dict.get("markdown_content")
        state.current_chapter_index = state_dict.get("current_chapter_index", 0)
        state.conversion_qa_passed = state_dict.get("conversion_qa_passed", False)
        state.conversion_qa_attempts = state_dict.get("conversion_qa_attempts", 0)
        state.split_qa_passed = state_dict.get("split_qa_passed", False)
        state.split_qa_attempts = state_dict.get("split_qa_attempts", 0)
        state.audio_qa_passed = state_dict.get("audio_qa_passed", False)
        state.audio_qa_attempts = state_dict.get("audio_qa_attempts", 0)
        state.total_chunks = state_dict.get("total_chunks", 0)
        state.processed_chunks = state_dict.get("processed_chunks", 0)
        state.errors = state_dict.get("errors", [])
        state.warnings = state_dict.get("warnings", [])
        state.book_title = state_dict.get("book_title")
        state.book_author = state_dict.get("book_author")
        state.chunk_size = state_dict.get("chunk_size", 500)
        state.workers = state_dict.get("workers", 1)

        # Restore chapters
        chapters_data = state_dict.get("chapters", [])
        state.chapters = [Chapter.from_dict(ch) for ch in chapters_data]

        return state

    def get_progress(self) -> dict:
        """Get current progress information."""
        return {
            "stage": self.stage.value,
            "chapters_total": len(self.chapters),
            "chapters_processed": sum(1 for ch in self.chapters if ch.audio_file),
            "chunks_total": self.total_chunks,
            "chunks_processed": self.processed_chunks,
            "current_chapter": self.current_chapter_index,
            "qa_status": {
                "conversion": self.conversion_qa_passed,
                "split": self.split_qa_passed,
                "audio": self.audio_qa_passed,
            }
        }
