"""End-to-end orchestration for retrieval, Gemini inference, ranking, and logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gemini_client import GeminiClient
from .recommender import load_songs, recommend_songs
from .retrieval import MusicKnowledgeBase, RetrievedSnippet


@dataclass
class InferredListenerProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    request_summary: str
    model_confidence: float
    uncertainty_notes: List[str] = field(default_factory=list)
    retrieval_used: List[str] = field(default_factory=list)
    catalog_fit: str = ""
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class RecommendationCandidate:
    title: str
    artist: str
    genre: str
    mood: str
    score: float
    explanation: str


@dataclass
class AdvisorResponse:
    prompt: str
    retrieved_context: List[RetrievedSnippet]
    inferred_profile: InferredListenerProfile
    recommendations: List[RecommendationCandidate]
    grounded_explanation: str
    confidence: float
    warnings: List[str]
    baseline_preferences: Dict[str, Any]
    baseline_top_title: Optional[str]
    used_fallback: bool = False
    log_path: Optional[str] = None


class MusicAdvisor:
    """Shared pipeline used by the CLI, evaluation harness, and Streamlit UI."""

    def __init__(
        self,
        songs: List[Dict[str, Any]],
        knowledge_base: MusicKnowledgeBase,
        llm_client: GeminiClient,
        log_path: str | Path = "logs/music_advisor_runs.jsonl",
    ):
        self.songs = songs
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client
        self.log_path = Path(log_path)

    def recommend_from_prompt(self, prompt: str, top_k: int = 5) -> AdvisorResponse:
        retrieved = self.knowledge_base.retrieve_context(prompt, top_k=3)
        retrieved_payload = [asdict(item) for item in retrieved]
        baseline_preferences = self._baseline_preferences(prompt)
        baseline_results = recommend_songs(baseline_preferences, self.songs, k=top_k)
        baseline_top_title = baseline_results[0][0]["title"] if baseline_results else None

        warnings: List[str] = []
        used_fallback = False

        try:
            raw_profile = self.llm_client.infer_preferences_with_gemini(
                prompt=prompt,
                retrieved_snippets=retrieved_payload,
                catalog_overview=self._catalog_overview(),
            )
            inferred_profile = self._normalize_profile(raw_profile)
        except Exception as exc:
            warnings.append(f"Gemini inference failed: {exc}")
            inferred_profile = self._fallback_profile(prompt, retrieved)
            used_fallback = True

        ranking_preferences = {
            "favorite_genre": inferred_profile.favorite_genre,
            "favorite_mood": inferred_profile.favorite_mood,
            "target_energy": inferred_profile.target_energy,
            "likes_acoustic": inferred_profile.likes_acoustic,
        }
        ranking_results = recommend_songs(ranking_preferences, self.songs, k=top_k)
        recommendations = [
            RecommendationCandidate(
                title=song["title"],
                artist=song["artist"],
                genre=song["genre"],
                mood=song["mood"],
                score=score,
                explanation=explanation,
            )
            for song, score, explanation in ranking_results
        ]

        if not retrieved:
            warnings.append("No guidance documents matched the request strongly; confidence may be lower.")
        if inferred_profile.uncertainty_notes:
            warnings.extend(inferred_profile.uncertainty_notes)

        confidence = self._compute_confidence(retrieved, inferred_profile, recommendations, used_fallback)

        grounded_explanation = self._build_grounded_explanation(
            prompt=prompt,
            retrieved_payload=retrieved_payload,
            inferred_profile=inferred_profile,
            ranking_results=ranking_results,
            warnings=warnings,
            used_fallback=used_fallback,
        )

        response = AdvisorResponse(
            prompt=prompt,
            retrieved_context=retrieved,
            inferred_profile=inferred_profile,
            recommendations=recommendations,
            grounded_explanation=grounded_explanation,
            confidence=confidence,
            warnings=warnings,
            baseline_preferences=baseline_preferences,
            baseline_top_title=baseline_top_title,
            used_fallback=used_fallback,
        )
        response.log_path = str(self._log_result(response))
        return response

    def _catalog_overview(self) -> Dict[str, List[str]]:
        genres = sorted({str(song["genre"]).lower() for song in self.songs})
        moods = sorted({str(song["mood"]).lower() for song in self.songs})
        return {"genres": genres, "moods": moods}

    def _normalize_profile(self, raw_profile: Dict[str, Any]) -> InferredListenerProfile:
        catalog = self._catalog_overview()
        genre = str(raw_profile.get("favorite_genre", "")).lower().strip()
        mood = str(raw_profile.get("favorite_mood", "")).lower().strip()
        if genre and genre not in catalog["genres"]:
            genre = ""
        if mood and mood not in catalog["moods"]:
            mood = ""

        target_energy = raw_profile.get("target_energy", 0.5)
        try:
            target_energy = float(target_energy)
        except (TypeError, ValueError):
            target_energy = 0.5
        target_energy = max(0.0, min(1.0, target_energy))

        confidence = raw_profile.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return InferredListenerProfile(
            favorite_genre=genre,
            favorite_mood=mood,
            target_energy=target_energy,
            likes_acoustic=bool(raw_profile.get("likes_acoustic", False)),
            request_summary=str(raw_profile.get("request_summary", "Recommendation intent inferred from the prompt.")),
            model_confidence=confidence,
            uncertainty_notes=[str(item) for item in raw_profile.get("uncertainty_notes", [])],
            retrieval_used=[str(item) for item in raw_profile.get("retrieval_used", [])],
            catalog_fit=str(raw_profile.get("catalog_fit", "")),
            reasoning_trace=[str(item) for item in raw_profile.get("reasoning_trace", [])],
        )

    def _fallback_profile(self, prompt: str, retrieved: List[RetrievedSnippet]) -> InferredListenerProfile:
        baseline = self._baseline_preferences(prompt)
        return InferredListenerProfile(
            favorite_genre=str(baseline["favorite_genre"]),
            favorite_mood=str(baseline["favorite_mood"]),
            target_energy=float(baseline["target_energy"]),
            likes_acoustic=bool(baseline["likes_acoustic"]),
            request_summary="Fallback heuristic profile because Gemini output was unavailable.",
            model_confidence=0.35,
            uncertainty_notes=["Fallback heuristic parsing was used instead of Gemini inference."],
            retrieval_used=[item.source for item in retrieved],
            catalog_fit="Best-effort fallback using keyword matches only.",
            reasoning_trace=["Matched direct keywords from the prompt to existing catalog labels."],
        )

    def _baseline_preferences(self, prompt: str) -> Dict[str, Any]:
        text = prompt.lower()
        genre_aliases = {
            "indie pop": "indie pop",
            "pop": "pop",
            "lofi": "lofi",
            "lo-fi": "lofi",
            "rock": "rock",
            "metal": "metal",
            "jazz": "jazz",
            "ambient": "ambient",
            "synthwave": "synthwave",
            "country": "country",
            "classical": "classical",
            "reggae": "reggae",
            "hip hop": "hip hop",
            "hip-hop": "hip hop",
            "folk": "folk",
            "punk": "punk",
        }
        mood_aliases = {
            "happy": "happy",
            "upbeat": "happy",
            "study": "focused",
            "focus": "focused",
            "coding": "focused",
            "chill": "chill",
            "relax": "relaxed",
            "relaxed": "relaxed",
            "moody": "moody",
            "sad": "sad",
            "intense": "intense",
            "workout": "intense",
        }

        favorite_genre = ""
        favorite_mood = ""
        for alias, canonical in genre_aliases.items():
            if alias in text:
                favorite_genre = canonical
                break
        for alias, canonical in mood_aliases.items():
            if alias in text:
                favorite_mood = canonical
                break

        target_energy = 0.55
        if any(token in text for token in ["workout", "gym", "high energy", "energetic", "hype"]):
            target_energy = 0.9
        elif any(token in text for token in ["study", "focus", "coding", "calm", "sleep"]):
            target_energy = 0.35
        elif any(token in text for token in ["chill", "relax", "easygoing"]):
            target_energy = 0.45

        likes_acoustic = any(token in text for token in ["acoustic", "organic", "unplugged", "folk"])

        return {
            "favorite_genre": favorite_genre,
            "favorite_mood": favorite_mood,
            "target_energy": target_energy,
            "likes_acoustic": likes_acoustic,
        }

    def _compute_confidence(
        self,
        retrieved: List[RetrievedSnippet],
        profile: InferredListenerProfile,
        recommendations: List[RecommendationCandidate],
        used_fallback: bool,
    ) -> float:
        retrieval_strength = min(1.0, len(retrieved) / 3.0)
        completeness = sum(
            [
                1.0 if profile.favorite_genre else 0.0,
                1.0 if profile.favorite_mood else 0.0,
                1.0 if 0.0 <= profile.target_energy <= 1.0 else 0.0,
                1.0,
            ]
        ) / 4.0

        top_alignment = 0.0
        if recommendations:
            top = recommendations[0]
            if profile.favorite_genre and top.genre == profile.favorite_genre:
                top_alignment += 0.35
            if profile.favorite_mood and top.mood == profile.favorite_mood:
                top_alignment += 0.25
            energy_gap = abs(profile.target_energy - self._song_energy(top.title))
            top_alignment += max(0.0, 0.4 - energy_gap) / 0.4 * 0.4

        score_gap = 0.0
        if len(recommendations) >= 2:
            score_gap = min(1.0, max(0.0, (recommendations[0].score - recommendations[1].score) / 2.0))
        elif recommendations:
            score_gap = 1.0

        confidence = (
            0.25 * retrieval_strength
            + 0.20 * completeness
            + 0.30 * top_alignment
            + 0.15 * score_gap
            + 0.10 * profile.model_confidence
        )
        if used_fallback:
            confidence -= 0.2
        if profile.uncertainty_notes:
            confidence -= min(0.15, 0.05 * len(profile.uncertainty_notes))
        return round(max(0.0, min(1.0, confidence)), 2)

    def _song_energy(self, title: str) -> float:
        for song in self.songs:
            if song["title"] == title:
                return float(song["energy"])
        return 0.5

    def _build_grounded_explanation(
        self,
        prompt: str,
        retrieved_payload: List[Dict[str, Any]],
        inferred_profile: InferredListenerProfile,
        ranking_results: List[Any],
        warnings: List[str],
        used_fallback: bool,
    ) -> str:
        if not ranking_results:
            return "No songs were available to rank."

        top_song, top_score, reasons = ranking_results[0]
        if used_fallback:
            explanation = (
                f"I used fallback keyword parsing for '{prompt}'. "
                f"The best match was {top_song['title']} by {top_song['artist']} because {reasons}."
            )
            if warnings:
                explanation += f" Warnings: {' '.join(warnings[:2])}"
            return explanation

        try:
            return self.llm_client.explain_recommendation(
                prompt=prompt,
                retrieved_snippets=retrieved_payload,
                inferred_preferences=asdict(inferred_profile),
                top_song=top_song,
                top_score=top_score,
                reasons=reasons,
            )
        except Exception as exc:
            warnings.append(f"Gemini explanation failed: {exc}")
            source_text = ", ".join(inferred_profile.retrieval_used) if inferred_profile.retrieval_used else "local scoring rules"
            return (
                f"{top_song['title']} by {top_song['artist']} ranked first because {reasons}. "
                f"The inference was grounded in {source_text}."
            )

    def _log_result(self, response: AdvisorResponse) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": response.prompt,
            "retrieved_context": [asdict(item) for item in response.retrieved_context],
            "inferred_profile": asdict(response.inferred_profile),
            "recommendations": [asdict(item) for item in response.recommendations],
            "grounded_explanation": response.grounded_explanation,
            "confidence": response.confidence,
            "warnings": response.warnings,
            "baseline_preferences": response.baseline_preferences,
            "baseline_top_title": response.baseline_top_title,
            "used_fallback": response.used_fallback,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return self.log_path


def build_default_advisor(
    data_path: str | Path = "data/songs.csv",
    docs_path: str | Path = "docs",
    log_path: str | Path = "logs/music_advisor_runs.jsonl",
) -> MusicAdvisor:
    """Create the production advisor wired to the local corpus and Gemini."""
    songs = load_songs(str(data_path))
    knowledge_base = MusicKnowledgeBase(docs_path)
    llm_client = GeminiClient()
    return MusicAdvisor(songs=songs, knowledge_base=knowledge_base, llm_client=llm_client, log_path=log_path)
