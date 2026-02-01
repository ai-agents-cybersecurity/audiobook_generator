"""
Chapter QA Agent - validates chapter splitting quality.
"""

from typing import Optional
from rich.console import Console

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import AudiobookState
from ..config import config, strip_thinking_tags, extract_json_from_response

console = Console()


class ChapterQAAgent:
    """
    QA Agent that validates chapter splitting quality.

    This agent checks:
    - Reasonable number of chapters
    - Consistent chapter sizes
    - Proper chapter titles
    - No missing content between chapters
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
        self.system_prompt = """You are a book structure QA specialist. Your task is to validate
        how a book has been split into chapters.

        Evaluate the following aspects:
        1. Chapter count - Is the number of chapters reasonable for the content length?
        2. Chapter sizes - Are chapters roughly balanced (not wildly different sizes)?
        3. Chapter titles - Do they look like proper chapter titles?
        4. Content coverage - Does the total chapter content account for the full book?

        Respond with a JSON object containing:
        {
            "passed": true/false,
            "issues": ["list of issues found"],
            "suggestions": ["list of improvements"],
            "recommended_action": "proceed" or "resplit" or "manual_review"
        }

        A book can have anywhere from 1 to 100+ chapters. Very short books might have just a few.
        Chapter size variation of 2-3x is acceptable. Larger variations warrant a note but not failure."""

    def analyze(self, state: AudiobookState) -> dict:
        """
        Analyze chapter split quality.

        Args:
            state: Workflow state with chapters

        Returns:
            Analysis result dict
        """
        if not state.chapters:
            return {
                "passed": False,
                "issues": ["No chapters found"],
                "suggestions": [],
                "recommended_action": "resplit",
            }

        # Calculate statistics
        total_content_len = len(state.markdown_content) if state.markdown_content else 0
        chapter_lens = [len(ch.content) for ch in state.chapters]
        total_chapter_len = sum(chapter_lens)
        avg_chapter_len = total_chapter_len // len(state.chapters) if state.chapters else 0

        # Prepare analysis prompt
        chapter_summary = "\n".join([
            f"  {i+1}. '{ch.title}' ({len(ch.content)} chars)"
            for i, ch in enumerate(state.chapters[:20])  # First 20 chapters
        ])

        if len(state.chapters) > 20:
            chapter_summary += f"\n  ... and {len(state.chapters) - 20} more chapters"

        prompt = f"""Analyze this chapter split:

        **Statistics:**
        - Total chapters: {len(state.chapters)}
        - Original content length: {total_content_len} chars
        - Total chapter content: {total_chapter_len} chars
        - Average chapter size: {avg_chapter_len} chars
        - Smallest chapter: {min(chapter_lens)} chars
        - Largest chapter: {max(chapter_lens)} chars
        - Size ratio (max/min): {max(chapter_lens) / max(min(chapter_lens), 1):.1f}x

        **Chapters:**
        {chapter_summary}

        Evaluate the split quality and respond with a JSON object."""

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
                # Fallback: analyze cleaned content
                cleaned = strip_thinking_tags(content)
                return {
                    "passed": "fail" not in cleaned.lower(),
                    "issues": [],
                    "suggestions": [],
                    "recommended_action": "proceed",
                }

        except Exception as e:
            console.print(f"[yellow]QA analysis warning: {e}[/yellow]")
            return self._basic_check(state)

    def _basic_check(self, state: AudiobookState) -> dict:
        """Perform basic quality checks without LLM."""
        issues = []
        suggestions = []

        chapters = state.chapters
        chapter_lens = [len(ch.content) for ch in chapters]

        # Check minimum chapters
        if len(chapters) == 0:
            issues.append("No chapters detected")
            return {
                "passed": False,
                "issues": issues,
                "suggestions": ["Check chapter detection patterns"],
                "recommended_action": "resplit",
            }

        # Check for very unbalanced chapters
        if chapter_lens:
            ratio = max(chapter_lens) / max(min(chapter_lens), 1)
            if ratio > 10:
                issues.append(f"Large chapter size imbalance: {ratio:.1f}x difference")
                suggestions.append("Consider more consistent chapter boundaries")

        # Check for very small chapters
        small_chapters = [i+1 for i, l in enumerate(chapter_lens) if l < 100]
        if small_chapters:
            issues.append(f"Very small chapters detected: {small_chapters}")
            suggestions.append("Consider merging small chapters")

        # Check chapter titles
        empty_titles = [i+1 for i, ch in enumerate(chapters) if not ch.title.strip()]
        if empty_titles:
            suggestions.append(f"Chapters {empty_titles} have empty titles")

        # Check content coverage
        total_original = len(state.markdown_content) if state.markdown_content else 0
        total_chapters = sum(chapter_lens)
        if total_original > 0:
            coverage = total_chapters / total_original
            if coverage < 0.9:
                issues.append(f"Only {coverage:.0%} of content in chapters - possible content loss")

        passed = len(issues) == 0
        action = "proceed" if passed else ("resplit" if any("content loss" in i for i in issues) else "manual_review")

        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
            "recommended_action": action,
        }

    def validate(self, state: AudiobookState) -> AudiobookState:
        """
        Run QA validation on the chapter split.

        Args:
            state: Workflow state

        Returns:
            Updated state with QA results
        """
        console.print("\n[cyan]Running chapter split QA...[/cyan]")

        state.split_qa_attempts += 1
        result = self.analyze(state)

        state.split_qa_passed = result["passed"]
        state.split_qa_issues = result.get("issues", [])

        if result["passed"]:
            console.print(f"[green]✓ Chapter split QA passed[/green]")
            console.print(f"  {len(state.chapters)} chapters detected")
        else:
            console.print(f"[yellow]⚠ Chapter split QA found issues:[/yellow]")
            for issue in result.get("issues", []):
                console.print(f"  - {issue}")

        if result.get("suggestions"):
            console.print("[blue]Suggestions:[/blue]")
            for suggestion in result["suggestions"]:
                console.print(f"  - {suggestion}")

        action = result.get("recommended_action", "proceed")
        if action != "proceed":
            console.print(f"[yellow]Recommended action: {action}[/yellow]")

        return state
