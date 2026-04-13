"""Gemini client wrapper for the Applied AI Music Advisor."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    genai = None


DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def _load_local_env(env_path: str | Path = ".env") -> None:
    """Populate os.environ from a simple local .env file when present."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class GeminiClient:
    """Wrapper around Gemini with prompts tailored to grounded music advice."""

    def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_MODEL_NAME):
        _load_local_env()
        if genai is None:
            raise RuntimeError(
                "The google-generativeai package is not installed. Run `pip install -r requirements.txt` first."
            )
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Set your Google AI Studio API key before running the advisor."
            )
        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def infer_preferences_with_gemini(
        self,
        prompt: str,
        retrieved_snippets: List[Dict[str, str]],
        catalog_overview: Dict[str, List[str]],
    ) -> Dict[str, object]:
        """Infer a structured preference profile grounded in retrieved context."""
        context_blocks = []
        for snippet in retrieved_snippets:
            context_blocks.append(
                f"Source: {snippet['source']}\nSnippet: {snippet['text']}\nMatched: {', '.join(snippet['matched_terms'])}"
            )

        prompt_text = f"""
You are an AI music recommendation analyst.

You will receive:
- a user request
- local retrieved guidance snippets
- the allowed genre and mood labels from the catalog

Your job:
- infer a structured listener profile for ranking songs
- stay grounded in the retrieved snippets and allowed catalog labels
- say when the request is only partially supported by the catalog

Allowed genres:
{", ".join(catalog_overview["genres"])}

Allowed moods:
{", ".join(catalog_overview["moods"])}

Retrieved guidance:
{chr(10).join(context_blocks) if context_blocks else "No snippets retrieved."}

User request:
{prompt}

Return JSON only. Use this schema exactly:
{{
  "favorite_genre": "string or empty string",
  "favorite_mood": "string or empty string",
  "target_energy": 0.0,
  "likes_acoustic": true,
  "request_summary": "one sentence",
  "confidence": 0.0,
  "uncertainty_notes": ["list of short notes"],
  "retrieval_used": ["source filenames you relied on"],
  "catalog_fit": "short sentence",
  "reasoning_trace": ["2-4 short bullets about how the request maps to the ranking fields"]
}}

Rules:
- favorite_genre must be one of the allowed genres or empty string.
- favorite_mood must be one of the allowed moods or empty string.
- target_energy must be between 0.0 and 1.0.
- If the request implies acoustic, set likes_acoustic true; otherwise false.
- If the request conflicts with the catalog, keep the best-fit labels and note the mismatch in uncertainty_notes.
- Do not invent labels outside the allowed lists.
"""
        response = self.model.generate_content(prompt_text)
        return self._parse_json_object((response.text or "").strip())

    def explain_recommendation(
        self,
        prompt: str,
        retrieved_snippets: List[Dict[str, str]],
        inferred_preferences: Dict[str, object],
        top_song: Dict[str, object],
        top_score: float,
        reasons: str,
    ) -> str:
        """Generate a grounded user-facing explanation for the top recommendation."""
        context_blocks = []
        for snippet in retrieved_snippets:
            context_blocks.append(f"{snippet['source']}: {snippet['text']}")

        explanation_prompt = f"""
You are explaining a music recommendation.

User request:
{prompt}

Retrieved guidance:
{chr(10).join(context_blocks) if context_blocks else "No snippets retrieved."}

Inferred preferences:
{json.dumps(inferred_preferences, indent=2)}

Top recommendation:
{json.dumps(top_song, indent=2)}

Score:
{top_score:.2f}

Ranking reasons:
{reasons}

Write 2-3 sentences.
Requirements:
- Mention at least one retrieved source filename if guidance was used.
- Tie the explanation to the song attributes.
- Mention uncertainty briefly if the request was only partially satisfied.
"""
        response = self.model.generate_content(explanation_prompt)
        return (response.text or "").strip()

    def _parse_json_object(self, text: str) -> Dict[str, object]:
        """Extract and parse JSON even if Gemini wraps it in fences."""
        if not text:
            raise ValueError("Gemini returned empty text.")

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Gemini did not return a JSON object.")

        return json.loads(text[start : end + 1])
