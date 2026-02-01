"""
TTS preprocessing node - generates per-chunk instructions for Qwen3-TTS.

Qwen3-TTS uses natural language instructions (via the 'instruct' parameter)
to control speech delivery, not inline SSML-style tags. This module analyzes
text chunks and generates appropriate instructions for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..config import config, extract_json_from_response, strip_thinking_tags
from ..state import AudiobookState, WorkflowStage, TTSChunk

console = Console()


@dataclass
class TTSInstructConfig:
    """Configuration for TTS instruction generation."""
    
    default_instruct: str = "Read in a clear, engaging audiobook narration style."
    scene_instructions: dict[str, str] = field(default_factory=dict)
    emotion_modifiers: dict[str, str] = field(default_factory=dict)
    character_voice_hints: dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def load(cls, path: str) -> "TTSInstructConfig":
        """Load configuration from JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            console.print(f"[yellow]TTS instruction config not found: {config_path}[/yellow]")
            return cls()
        
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return cls(
                default_instruct=data.get("default_instruct", cls.default_instruct),
                scene_instructions=data.get("scene_instructions", {}),
                emotion_modifiers=data.get("emotion_modifiers", {}),
                character_voice_hints=data.get("character_voice_hints", {}),
            )
        except Exception as e:
            console.print(f"[yellow]Error loading TTS config: {e}[/yellow]")
            return cls()
    
    def has_config(self) -> bool:
        """Check if meaningful configuration is loaded."""
        return bool(self.scene_instructions or self.emotion_modifiers)
    
    def build_prompt_context(self) -> str:
        """Build context string for LLM prompts."""
        lines = ["Available scene types and their base instructions:"]
        for scene, instr in self.scene_instructions.items():
            lines.append(f"  - {scene}: {instr}")
        
        lines.append("\nAvailable emotion modifiers:")
        for emotion, modifier in self.emotion_modifiers.items():
            lines.append(f"  - {emotion}: {modifier}")
        
        if self.character_voice_hints:
            lines.append("\nCharacter voice hints:")
            for char_type, hint in self.character_voice_hints.items():
                lines.append(f"  - {char_type}: {hint}")
        
        return "\n".join(lines)


@dataclass
class ChunkAnalysisState:
    """State for chunk analysis subgraph."""
    chunk_text: str
    chapter_title: str
    chapter_number: int
    chapter_context: str
    config_context: str
    default_instruct: str
    
    # Output
    instruct: Optional[str] = None
    scene_type: Optional[str] = None
    emotion: Optional[str] = None
    notes: Optional[str] = None


def build_instruction_graph(llm: ChatOllama, tts_config: TTSInstructConfig):
    """Build the LangGraph for generating TTS instructions per chunk."""
    
    def analyze_chunk(state: ChunkAnalysisState) -> ChunkAnalysisState:
        """Analyze chunk and generate appropriate TTS instruction."""
        
        system_prompt = """You are an expert audiobook director. Your task is to analyze text chunks and generate natural language instructions for a text-to-speech system.

The TTS system (Qwen3-TTS) uses an 'instruct' parameter that accepts natural language descriptions of how to speak, such as:
- "Read with excitement and energy"
- "Speak softly and contemplatively"  
- "Read as an angry older man arguing"
- "Speak with tension and urgency, as if in danger"

Analyze the provided text chunk and determine the best instruction based on:
1. Scene type (dialogue, action, reflection, narration, description)
2. Emotional tone (angry, sad, excited, tense, calm, etc.)
3. If dialogue, any character voice hints

Return JSON with:
{
  "scene_type": "dialogue|action|reflection|narration|description|technical|emotional",
  "emotion": "the dominant emotion or 'neutral'",
  "instruct": "A concise natural language instruction for the TTS (1-2 sentences max)",
  "notes": "Brief explanation of your choice"
}

Keep instructions concise but evocative. The instruction should be practical for voice acting."""

        user_prompt = f"""Chapter {state.chapter_number}: {state.chapter_title}

Chapter Context:
{state.chapter_context}

Available Configuration:
{state.config_context}

---
Text Chunk to Analyze:
{state.chunk_text}
---

Generate the TTS instruction for this chunk."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            result = extract_json_from_response(response.content)
            
            if result and isinstance(result, dict):
                state.scene_type = result.get("scene_type", "narration")
                state.emotion = result.get("emotion", "neutral")
                state.instruct = result.get("instruct", state.default_instruct)
                state.notes = result.get("notes", "")
            else:
                # Fallback: try to extract just the instruction from text
                cleaned = strip_thinking_tags(response.content)
                if cleaned and len(cleaned) < 200:
                    state.instruct = cleaned.strip()
                else:
                    state.instruct = state.default_instruct
                state.scene_type = "narration"
                state.emotion = "neutral"
                
        except Exception as e:
            console.print(f"[yellow]Chunk analysis error: {e}[/yellow]")
            state.instruct = state.default_instruct
            state.scene_type = "narration"
            state.emotion = "neutral"
        
        return state
    
    def validate_instruction(state: ChunkAnalysisState) -> ChunkAnalysisState:
        """Validate and clean up the generated instruction."""
        
        if not state.instruct or len(state.instruct) < 5:
            state.instruct = state.default_instruct
        
        # Ensure instruction isn't too long (TTS works better with concise instructions)
        if len(state.instruct) > 200:
            state.instruct = state.instruct[:200].rsplit(' ', 1)[0] + "."
        
        # Clean up
        state.instruct = state.instruct.strip()
        if not state.instruct.endswith('.'):
            state.instruct += '.'
        
        return state
    
    graph = StateGraph(ChunkAnalysisState)
    graph.add_node("analyze", analyze_chunk)
    graph.add_node("validate", validate_instruction)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "validate")
    graph.add_edge("validate", END)
    
    return graph.compile()


def summarize_chapter_context(llm: ChatOllama, chapter_text: str, chapter_title: str, chapter_number: int) -> str:
    """Generate a brief context summary for the chapter."""
    
    # Trim text to avoid token limits
    max_chars = 6000
    if len(chapter_text) > max_chars:
        half = max_chars // 2
        excerpt = f"{chapter_text[:half].strip()}\n...\n{chapter_text[-half:].strip()}"
    else:
        excerpt = chapter_text
    
    system_prompt = """You are a literary analyst preparing context for audiobook narration.
Provide a brief summary focusing on:
1. Main characters and their relationships
2. Overall tone and mood
3. Key emotional beats

Return JSON with: {"summary": "...", "main_characters": [...], "overall_tone": "..."}
Keep the summary under 200 words."""

    user_prompt = f"""Chapter {chapter_number}: {chapter_title}

Excerpt:
{excerpt}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        result = extract_json_from_response(response.content)
        if result and isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return strip_thinking_tags(response.content) or "No context available."
    except Exception as e:
        console.print(f"[yellow]Context summary error: {e}[/yellow]")
        return "No context available."


def preprocess_tts(state: AudiobookState) -> AudiobookState:
    """LangGraph node: Generate per-chunk TTS instructions.
    
    This node analyzes each text chunk and generates an appropriate
    natural language instruction for Qwen3-TTS's 'instruct' parameter.
    """
    state.stage = WorkflowStage.PREPROCESSING
    
    if not state.chapters:
        state.errors.append("No chapters available for TTS preprocessing")
        state.stage = WorkflowStage.FAILED
        return state
    
    # Check if preprocessing is enabled
    if not config.TTS_PREPROCESSING_ENABLED:
        console.print("[yellow]TTS preprocessing disabled; using default instruction for all chunks.[/yellow]")
        default_instruct = "Read in a clear, engaging audiobook narration style."
        for chapter in state.chapters:
            chapter.tts_chunks = [
                TTSChunk(text=chunk, instruct=default_instruct)
                for chunk in chapter.chunks
            ]
        return state
    
    # Load TTS instruction configuration
    config_path = config.QWEN3_TTS_CONFIG_PATH if hasattr(config, 'QWEN3_TTS_CONFIG_PATH') else str(Path(__file__).parent.parent / "qwen3_tts_config.json")
    tts_config = TTSInstructConfig.load(config_path)
    
    if not tts_config.has_config():
        console.print("[yellow]No TTS instruction config found; using default instruction for all chunks.[/yellow]")
        for chapter in state.chapters:
            chapter.tts_chunks = [
                TTSChunk(text=chunk, instruct=tts_config.default_instruct)
                for chunk in chapter.chunks
            ]
        return state
    
    console.print("\n[bold blue]Step 5: Generating TTS instructions for chunks...[/bold blue]")
    console.print(f"[dim]Using config: {config_path}[/dim]")
    
    # Initialize LLM
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_predict=512,  # Instructions should be short
    )
    
    # Build the instruction generation graph
    instruction_graph = build_instruction_graph(llm, tts_config)
    config_context = tts_config.build_prompt_context()
    
    total_chunks = sum(len(ch.chunks) for ch in state.chapters)
    processed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating instructions...", total=total_chunks)
        
        for chapter in state.chapters:
            if not chapter.chunks:
                chapter.tts_chunks = []
                continue
            
            # Get chapter context (summarize once per chapter)
            chapter_text = chapter.cleaned_content or chapter.content
            chapter_context = summarize_chapter_context(
                llm,
                chapter_text=chapter_text or "",
                chapter_title=chapter.title,
                chapter_number=chapter.number,
            )
            
            tts_chunks: list[TTSChunk] = []
            
            for chunk in chapter.chunks:
                # Analyze chunk and generate instruction
                chunk_state = ChunkAnalysisState(
                    chunk_text=chunk,
                    chapter_title=chapter.title,
                    chapter_number=chapter.number,
                    chapter_context=chapter_context,
                    config_context=config_context,
                    default_instruct=tts_config.default_instruct,
                )
                
                try:
                    result = instruction_graph.invoke(chunk_state)
                    
                    if isinstance(result, dict):
                        instruct = result.get("instruct", tts_config.default_instruct)
                    else:
                        instruct = result.instruct or tts_config.default_instruct
                        
                except Exception as e:
                    state.warnings.append(f"Chapter {chapter.number} instruction error: {e}")
                    instruct = tts_config.default_instruct
                
                tts_chunks.append(TTSChunk(text=chunk, instruct=instruct))
                processed += 1
                progress.update(task, completed=processed)
            
            chapter.tts_chunks = tts_chunks
            console.print(f"  [green]✓ Chapter {chapter.number}: {len(tts_chunks)} chunks processed[/green]")
    
    console.print(f"[green]Generated instructions for {processed} chunks[/green]")
    
    return state
