"""Evaluation harness for the Gemini-RAG music advisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .music_advisor import MusicAdvisor


@dataclass
class EvaluationScenario:
    prompt: str
    expected_genre: str
    expected_mood: str
    expected_source: str


SCENARIOS = [
    EvaluationScenario(
        prompt="I need calm background music for studying and deep focus.",
        expected_genre="lofi",
        expected_mood="focused",
        expected_source="contexts.md",
    ),
    EvaluationScenario(
        prompt="Give me something intense and high energy for a workout.",
        expected_genre="rock",
        expected_mood="intense",
        expected_source="contexts.md",
    ),
    EvaluationScenario(
        prompt="I want acoustic and chill songs for winding down tonight.",
        expected_genre="folk",
        expected_mood="chill",
        expected_source="mood_energy_guide.md",
    ),
    EvaluationScenario(
        prompt="Recommend sad but energetic pop for a dramatic run.",
        expected_genre="pop",
        expected_mood="sad",
        expected_source="fallback_rules.md",
    ),
    EvaluationScenario(
        prompt="Play something unusual and futuristic for me.",
        expected_genre="",
        expected_mood="",
        expected_source="fallback_rules.md",
    ),
]


def run_evaluation(advisor: MusicAdvisor) -> Dict[str, object]:
    """Run benchmark prompts and summarize reliability signals."""
    detailed_results: List[Dict[str, object]] = []
    retrieval_hits = 0
    alignment_hits = 0
    low_confidence_cases = 0
    confidence_total = 0.0

    for scenario in SCENARIOS:
        response = advisor.recommend_from_prompt(scenario.prompt, top_k=3)
        retrieved_sources = [item.source for item in response.retrieved_context]
        retrieval_hit = scenario.expected_source in retrieved_sources
        if retrieval_hit:
            retrieval_hits += 1

        top = response.recommendations[0] if response.recommendations else None
        alignment_hit = False
        if top:
            genre_ok = not scenario.expected_genre or top.genre == scenario.expected_genre
            mood_ok = not scenario.expected_mood or top.mood == scenario.expected_mood
            alignment_hit = genre_ok or mood_ok
        if alignment_hit:
            alignment_hits += 1

        if response.confidence < 0.5:
            low_confidence_cases += 1
        confidence_total += response.confidence

        detailed_results.append(
            {
                "prompt": scenario.prompt,
                "retrieved_sources": retrieved_sources,
                "top_title": top.title if top else None,
                "top_genre": top.genre if top else None,
                "top_mood": top.mood if top else None,
                "confidence": response.confidence,
                "warnings": response.warnings,
                "retrieval_hit": retrieval_hit,
                "alignment_hit": alignment_hit,
                "used_fallback": response.used_fallback,
            }
        )

    total = len(SCENARIOS)
    return {
        "retrieval_hit_rate": round(retrieval_hits / total, 2),
        "alignment_rate": round(alignment_hits / total, 2),
        "average_confidence": round(confidence_total / total, 2),
        "low_confidence_cases": low_confidence_cases,
        "details": detailed_results,
    }


def print_evaluation(summary: Dict[str, object]) -> None:
    """Pretty-print evaluation results for CLI usage."""
    print("\nEvaluation Summary")
    print("------------------")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']:.2f}")
    print(f"Alignment rate: {summary['alignment_rate']:.2f}")
    print(f"Average confidence: {summary['average_confidence']:.2f}")
    print(f"Low-confidence cases: {summary['low_confidence_cases']}")
    print()
    for item in summary["details"]:
        print(f"Prompt: {item['prompt']}")
        print(f"  Retrieved: {item['retrieved_sources']}")
        print(f"  Top result: {item['top_title']} ({item['top_genre']}, {item['top_mood']})")
        print(f"  Confidence: {item['confidence']:.2f}")
        print(f"  Retrieval hit: {item['retrieval_hit']}")
        print(f"  Alignment hit: {item['alignment_hit']}")
        print(f"  Fallback used: {item['used_fallback']}")
        if item["warnings"]:
            print(f"  Warnings: {' | '.join(item['warnings'])}")
        print()
