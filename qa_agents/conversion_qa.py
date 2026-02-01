"""
Conversion QA Agent - validates document conversion quality.
"""

from typing import Optional
from rich.console import Console

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import AudiobookState
from ..config import config, strip_thinking_tags, extract_json_from_response

console = Console()


class ConversionQAAgent:
    """
    QA Agent that validates document conversion quality using LLM analysis.

    This agent checks:
    - Content completeness (no major text loss)
    - Structure preservation (headings, paragraphs)
    - Character encoding issues
    - Metadata extraction
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
        self.system_prompt = """You are a document conversion QA specialist. Your task is to analyze
        the quality of a document conversion from its original format to markdown.

        Evaluate the following aspects:
        1. Content completeness - Is all the text present?
        2. Structure preservation - Are headings, paragraphs, and sections maintained?
        3. Character issues - Are there encoding problems or garbled text?
        4. Readability - Is the converted text readable and properly formatted?

        Respond with a JSON object containing:
        {
            "passed": true/false,
            "issues": ["list of issues found"],
            "suggestions": ["list of improvements"],
            "confidence": 0.0-1.0
        }

        Be strict but fair. Minor formatting differences are acceptable.
        Major content loss or unreadable sections should fail."""

    def analyze(self, state: AudiobookState) -> dict:
        """
        Analyze conversion quality.

        Args:
            state: Workflow state with converted content

        Returns:
            Analysis result dict
        """
        if not state.markdown_content:
            return {
                "passed": False,
                "issues": ["No markdown content to analyze"],
                "suggestions": [],
                "confidence": 1.0,
            }

        # Prepare analysis prompt
        content_sample = state.markdown_content[:3000]  # First 3000 chars
        content_end = state.markdown_content[-1000:] if len(state.markdown_content) > 4000 else ""

        prompt = f"""Analyze this document conversion:

        **Metadata:**
        - Title: {state.book_title or 'Not detected'}
        - Author: {state.book_author or 'Not detected'}
        - Total characters: {len(state.markdown_content)}

        **Content Sample (first 3000 chars):**
        ```
        {content_sample}
        ```

        **Content End (last 1000 chars):**
        ```
        {content_end}
        ```

        Evaluate the conversion quality and respond with a JSON object."""

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
                    "passed": "fail" not in cleaned.lower() and "error" not in cleaned.lower(),
                    "issues": [],
                    "suggestions": [],
                    "confidence": 0.7,
                }

        except Exception as e:
            console.print(f"[yellow]QA analysis warning: {e}[/yellow]")
            # Fallback to basic checks
            return self._basic_check(state)

    def _basic_check(self, state: AudiobookState) -> dict:
        """Perform basic quality checks without LLM."""
        issues = []
        suggestions = []

        content = state.markdown_content

        # Check for minimum content
        if len(content) < 1000:
            issues.append("Very short content - possible conversion failure")

        # Check for garbled text patterns
        import re
        garbled_patterns = [
            r"[\x00-\x08\x0b\x0c\x0e-\x1f]",  # Control characters
            r"\\[a-z]+\d+",  # RTF artifacts
            r"[^\x00-\x7F]{10,}",  # Long non-ASCII sequences
        ]

        for pattern in garbled_patterns:
            if re.search(pattern, content):
                issues.append(f"Potential encoding issue detected (pattern: {pattern})")

        # Check for structure
        if "\n\n" not in content:
            issues.append("No paragraph breaks detected")
            suggestions.append("Check if paragraphs were properly preserved")

        # Check for headings
        if not re.search(r"^#+\s", content, re.MULTILINE):
            suggestions.append("No markdown headings found - consider adding chapter markers")

        passed = len(issues) == 0

        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.8 if passed else 0.6,
        }

    def validate(self, state: AudiobookState) -> AudiobookState:
        """
        Run QA validation on the conversion.

        Args:
            state: Workflow state

        Returns:
            Updated state with QA results
        """
        console.print("\n[cyan]Running conversion QA...[/cyan]")

        state.conversion_qa_attempts += 1
        result = self.analyze(state)

        state.conversion_qa_passed = result["passed"]
        state.conversion_qa_issues = result.get("issues", [])

        if result["passed"]:
            console.print(f"[green]✓ Conversion QA passed[/green] (confidence: {result.get('confidence', 0):.0%})")
        else:
            console.print(f"[yellow]⚠ Conversion QA found issues:[/yellow]")
            for issue in result.get("issues", []):
                console.print(f"  - {issue}")

        if result.get("suggestions"):
            console.print("[blue]Suggestions:[/blue]")
            for suggestion in result["suggestions"]:
                console.print(f"  - {suggestion}")

        return state
