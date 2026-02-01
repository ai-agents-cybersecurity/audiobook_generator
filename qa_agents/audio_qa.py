"""
Audio QA Agent - validates audio generation quality.
"""

from typing import Optional
from rich.console import Console

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import AudiobookState
from ..config import config, strip_thinking_tags, extract_json_from_response

console = Console()


class AudioQAAgent:
    """
    QA Agent that validates audio generation quality.

    This agent reviews:
    - Verification scores from STT comparison
    - Audio file existence and size
    - Overall generation success rate
    """

    def __init__(self, model_name: str = None):
        # Use config if no model specified
        if model_name is None:
            model_name = config.OLLAMA_MODEL
        console.print(f"[dim]Using Ollama model: {model_name} (ctx={config.OLLAMA_NUM_CTX})[/dim]")
        self.llm = ChatOllama(
            model=model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=config.OLLAMA_NUM_CTX,
            num_predict=config.OLLAMA_NUM_PREDICT,
        )
        self.system_prompt = """You are an audiobook quality assurance specialist. Your task is to
        evaluate the quality of generated audiobook chapters based on verification metrics.

        Evaluate the following aspects:
        1. Verification scores - Are the STT transcription similarities acceptable?
        2. Coverage - Were all chapters successfully generated?
        3. Consistency - Are verification scores consistent across chapters?

        Respond with a JSON object containing:
        {
            "passed": true/false,
            "issues": ["list of issues found"],
            "suggestions": ["list of improvements"],
            "chapters_to_regenerate": [list of chapter numbers that should be regenerated],
            "overall_quality": "excellent" or "good" or "acceptable" or "poor"
        }

        A similarity score of 60% or higher is generally acceptable for audiobooks.
        Lower scores might indicate pronunciation issues or text mismatches."""

    def analyze(self, state: AudiobookState) -> dict:
        """
        Analyze audio generation quality.

        Args:
            state: Workflow state with generated audio

        Returns:
            Analysis result dict
        """
        if not state.chapters:
            return {
                "passed": False,
                "issues": ["No chapters to analyze"],
                "suggestions": [],
                "chapters_to_regenerate": [],
                "overall_quality": "poor",
            }

        # Gather statistics
        chapters_with_audio = [ch for ch in state.chapters if ch.audio_file]
        verified_chapters = [ch for ch in chapters_with_audio if ch.verified]
        scores = [ch.verification_score for ch in chapters_with_audio if ch.verification_score > 0]

        # Prepare analysis prompt
        chapter_results = []
        for ch in state.chapters:
            status = "✓" if ch.verified else ("⚠" if ch.audio_file else "✗")
            score = f"{ch.verification_score:.0%}" if ch.verification_score > 0 else "N/A"
            chapter_results.append(f"  {status} Chapter {ch.number}: {score}")

        # Calculate stats safely
        avg_score = f"{sum(scores) / len(scores):.0%}" if scores else "N/A"
        min_score = f"{min(scores):.0%}" if scores else "N/A"
        max_score = f"{max(scores):.0%}" if scores else "N/A"
        chapter_results_str = "\n".join(chapter_results[:30])
        more_chapters = f"\n... and {len(state.chapters) - 30} more" if len(state.chapters) > 30 else ""

        prompt = f"""Analyze these audiobook generation results:

        **Summary:**
        - Total chapters: {len(state.chapters)}
        - Chapters with audio: {len(chapters_with_audio)}
        - Chapters verified: {len(verified_chapters)}
        - Average similarity score: {avg_score}
        - Lowest score: {min_score}
        - Highest score: {max_score}

        **Chapter Results:**
        {chapter_results_str}{more_chapters}

        Evaluate the overall audio quality and respond with a JSON object."""

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)

            # Parse response (handles thinking model output)
            content = response.content

            # Extract JSON, stripping any <think> tags from thinking models
            result = extract_json_from_response(content)

            if result:
                return result
            else:
                return self._basic_check(state)

        except Exception as e:
            console.print(f"[yellow]QA analysis warning: {e}[/yellow]")
            return self._basic_check(state)

    def _basic_check(self, state: AudiobookState) -> dict:
        """Perform basic quality checks without LLM."""
        issues = []
        suggestions = []
        chapters_to_regenerate = []

        chapters_with_audio = [ch for ch in state.chapters if ch.audio_file]
        scores = [ch.verification_score for ch in chapters_with_audio if ch.verification_score > 0]

        # Check coverage
        coverage = len(chapters_with_audio) / len(state.chapters) if state.chapters else 0
        if coverage < 0.9:
            issues.append(f"Only {coverage:.0%} of chapters have audio")
            missing = [ch.number for ch in state.chapters if not ch.audio_file]
            chapters_to_regenerate.extend(missing)

        # Check verification scores
        low_score_chapters = [
            ch.number for ch in chapters_with_audio
            if ch.verification_score > 0 and ch.verification_score < 0.5
        ]
        if low_score_chapters:
            issues.append(f"Low verification scores for chapters: {low_score_chapters}")
            chapters_to_regenerate.extend(low_score_chapters)
            suggestions.append("Consider regenerating chapters with low scores")

        # Determine overall quality
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.8:
                quality = "excellent"
            elif avg_score >= 0.7:
                quality = "good"
            elif avg_score >= 0.6:
                quality = "acceptable"
            else:
                quality = "poor"
        else:
            quality = "unknown"

        passed = len(issues) == 0 and coverage >= 0.9

        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
            "chapters_to_regenerate": list(set(chapters_to_regenerate)),
            "overall_quality": quality,
        }

    def validate(self, state: AudiobookState) -> AudiobookState:
        """
        Run QA validation on the generated audio.

        Args:
            state: Workflow state

        Returns:
            Updated state with QA results
        """
        console.print("\n[cyan]Running audio QA...[/cyan]")

        state.audio_qa_attempts += 1
        result = self.analyze(state)

        state.audio_qa_passed = result["passed"]
        state.audio_qa_issues = result.get("issues", [])

        quality = result.get("overall_quality", "unknown")

        if result["passed"]:
            console.print(f"[green]✓ Audio QA passed[/green] (quality: {quality})")
        else:
            console.print(f"[yellow]⚠ Audio QA found issues:[/yellow]")
            for issue in result.get("issues", []):
                console.print(f"  - {issue}")

        if result.get("suggestions"):
            console.print("[blue]Suggestions:[/blue]")
            for suggestion in result["suggestions"]:
                console.print(f"  - {suggestion}")

        if result.get("chapters_to_regenerate"):
            console.print(f"[yellow]Chapters to consider regenerating: {result['chapters_to_regenerate']}[/yellow]")

        return state
