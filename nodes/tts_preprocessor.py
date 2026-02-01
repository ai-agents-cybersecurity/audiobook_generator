"""
TTS preprocessing node - enriches cleaned chunks with Qwen3-TTS tags.

This step uses a LangGraph subgraph with reflection to annotate dialogue,
speaker context, and delivery cues while enforcing a strict tag allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console

from ..config import config, extract_json_from_response, strip_thinking_tags
from ..state import AudiobookState, WorkflowStage

console = Console()

TAG_PATTERN = re.compile(r"<[^>]+>|\[[^\]]+\]")


@dataclass
class Qwen3TTSTagRegistry:
    """Registry of approved Qwen3-TTS tags for validation."""

    literal_tags: set[str] = field(default_factory=set)
    pattern_tags: list[re.Pattern] = field(default_factory=list)
    tag_descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Qwen3TTSTagRegistry":
        registry_path = Path(path)
        if not registry_path.exists():
            console.print(f"[yellow]Qwen3-TTS tag registry not found: {registry_path}[/yellow]")
            return cls()

        data = json.loads(registry_path.read_text(encoding="utf-8"))
        literal_tags = set(data.get("literal_tags", []))
        pattern_tags = [re.compile(p) for p in data.get("pattern_tags", [])]
        tag_descriptions = data.get("tag_descriptions", {})
        return cls(literal_tags=literal_tags, pattern_tags=pattern_tags, tag_descriptions=tag_descriptions)

    def has_tags(self) -> bool:
        return bool(self.literal_tags or self.pattern_tags)

    def is_valid_tag(self, tag: str) -> bool:
        if tag in self.literal_tags:
            return True
        return any(pattern.fullmatch(tag) for pattern in self.pattern_tags)

    def allowed_tags_for_prompt(self) -> str:
        lines = []
        for tag in sorted(self.literal_tags):
            description = self.tag_descriptions.get(tag, "")
            lines.append(f"- {tag}{' — ' + description if description else ''}")
        for pattern in self.pattern_tags:
            lines.append(f"- {pattern.pattern}")
        return "\n".join(lines) if lines else "(no tags configured)"


def sanitize_tagged_text(text: str, registry: Qwen3TTSTagRegistry) -> tuple[str, list[str]]:
    removed: list[str] = []

    def _replace(match: re.Match) -> str:
        tag = match.group(0)
        if registry.is_valid_tag(tag):
            return tag
        removed.append(tag)
        return ""

    sanitized = TAG_PATTERN.sub(_replace, text)
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
    return sanitized, removed


def trim_text(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half].strip()}\n...\n{text[-half:].strip()}"


@dataclass
class ChunkTagState:
    chunk_text: str
    chapter_title: str
    chapter_number: int
    chapter_context: str
    allowed_tags: str
    draft_text: Optional[str] = None
    annotated_text: Optional[str] = None
    issues: list[str] = field(default_factory=list)


def build_chunk_tag_graph(llm: ChatOllama, registry: Qwen3TTSTagRegistry):
    def node_draft(state: ChunkTagState) -> ChunkTagState:
        system_prompt = (
            "You are an expert audiobook director specializing in Qwen3-TTS tagging. "
            "Use ONLY the approved tags provided. Never invent new tags. "
            "Return JSON with fields: annotated_text, tags_used, notes."
        )
        prompt = (
            f"Chapter {state.chapter_number}: {state.chapter_title}\n"
            f"Chapter context summary:\n{state.chapter_context}\n\n"
            f"Allowed tags (exact or regex patterns):\n{state.allowed_tags}\n\n"
            "Annotate the chunk for speaker changes, emotion, pacing, and emphasis.\n"
            "Only insert tags where they improve delivery. Do not rewrite content.\n\n"
            f"Chunk:\n{state.chunk_text}"
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        result = extract_json_from_response(response.content)
        if result and isinstance(result, dict):
            state.draft_text = result.get("annotated_text", state.chunk_text)
        else:
            state.draft_text = strip_thinking_tags(response.content) or state.chunk_text
            state.issues.append("Draft agent returned non-JSON output")
        state.annotated_text = state.draft_text
        return state

    def node_reflect(state: ChunkTagState) -> ChunkTagState:
        system_prompt = (
            "You are a strict reviewer for Qwen3-TTS tag usage. "
            "Fix any incorrect or missing tags using only the approved list. "
            "Return JSON with fields: annotated_text, notes."
        )
        prompt = (
            f"Allowed tags:\n{state.allowed_tags}\n\n"
            f"Original chunk:\n{state.chunk_text}\n\n"
            f"Annotated chunk:\n{state.annotated_text}"
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        result = extract_json_from_response(response.content)
        if result and isinstance(result, dict):
            state.annotated_text = result.get("annotated_text", state.annotated_text)
        else:
            state.issues.append("Reflection agent returned non-JSON output")
        return state

    def node_validate(state: ChunkTagState) -> ChunkTagState:
        if not state.annotated_text:
            state.annotated_text = state.chunk_text
            state.issues.append("Annotation empty; reverted to original chunk")
            return state

        sanitized, removed = sanitize_tagged_text(state.annotated_text, registry)
        if removed:
            state.issues.append(f"Removed invalid tags: {', '.join(removed[:5])}")
        state.annotated_text = sanitized or state.chunk_text
        return state

    graph = StateGraph(ChunkTagState)
    graph.add_node("draft", node_draft)
    graph.add_node("reflect", node_reflect)
    graph.add_node("validate", node_validate)
    graph.set_entry_point("draft")
    graph.add_edge("draft", "reflect")
    graph.add_edge("reflect", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def summarize_chapter_context(llm: ChatOllama, chapter_text: str, chapter_title: str, chapter_number: int) -> str:
    excerpt = trim_text(chapter_text)
    system_prompt = (
        "You are a literary analyst preparing context for TTS tagging. "
        "Summarize speaker identities, tone shifts, and notable emotions. "
        "Return JSON with fields: speakers, narration_style, tone_notes."
    )
    prompt = (
        f"Chapter {chapter_number}: {chapter_title}\n"
        f"Excerpt:\n{excerpt}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    result = extract_json_from_response(response.content)
    if result and isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    cleaned = strip_thinking_tags(response.content)
    return cleaned or "No context summary available."


def preprocess_tts(state: AudiobookState) -> AudiobookState:
    """LangGraph node: add Qwen3-TTS tags to chunks before generation."""
    state.stage = WorkflowStage.PREPROCESSING

    if not state.chapters:
        state.errors.append("No chapters available for TTS preprocessing")
        state.stage = WorkflowStage.FAILED
        return state

    if not config.TTS_PREPROCESSING_ENABLED:
        console.print("[yellow]TTS preprocessing disabled; using cleaned chunks as-is.[/yellow]")
        for chapter in state.chapters:
            chapter.tts_chunks = list(chapter.chunks)
        return state

    registry = Qwen3TTSTagRegistry.load(config.QWEN3_TTS_TAGS_PATH)
    if not registry.has_tags():
        warning = "No Qwen3-TTS tags configured; skipping tag enrichment."
        console.print(f"[yellow]{warning}[/yellow]")
        state.warnings.append(warning)
        for chapter in state.chapters:
            chapter.tts_chunks = list(chapter.chunks)
        return state

    console.print("\n[bold blue]Step 5: Preprocessing text for Qwen3-TTS tags...[/bold blue]")
    console.print(f"[dim]Using Qwen3-TTS tag registry: {config.QWEN3_TTS_TAGS_PATH}[/dim]")

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_predict=config.OLLAMA_NUM_PREDICT,
    )

    chunk_graph = build_chunk_tag_graph(llm, registry)
    allowed_tags_prompt = registry.allowed_tags_for_prompt()

    for chapter in state.chapters:
        if not chapter.chunks:
            chapter.tts_chunks = []
            continue

        chapter_text = chapter.cleaned_content or chapter.content
        chapter_context = summarize_chapter_context(
            llm,
            chapter_text=chapter_text or "",
            chapter_title=chapter.title,
            chapter_number=chapter.number,
        )

        annotated_chunks: list[str] = []
        for chunk in chapter.chunks:
            chunk_state = ChunkTagState(
                chunk_text=chunk,
                chapter_title=chapter.title,
                chapter_number=chapter.number,
                chapter_context=chapter_context,
                allowed_tags=allowed_tags_prompt,
            )
            result = chunk_graph.invoke(chunk_state)
            if isinstance(result, dict):
                annotated = result.get("annotated_text", chunk)
                issues = result.get("issues", [])
            else:
                annotated = result.annotated_text or chunk
                issues = result.issues
            if issues:
                for issue in issues:
                    state.warnings.append(
                        f"Chapter {chapter.number} tag warning: {issue}"
                    )
            annotated_chunks.append(annotated or chunk)

        chapter.tts_chunks = annotated_chunks

    return state
