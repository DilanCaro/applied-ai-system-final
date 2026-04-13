"""CLI demo for the Gemini-RAG music advisor."""

from .evaluation import print_evaluation, run_evaluation
from .music_advisor import build_default_advisor


SHOWCASE_PROMPTS = [
    "I need calm background music for studying and deep focus.",
    "Give me something intense and high energy for a workout.",
    "Recommend sad but energetic pop for a dramatic run.",
]


def _print_response(prompt: str, advisor) -> None:
    response = advisor.recommend_from_prompt(prompt, top_k=5)
    print(f"\n{'=' * 72}")
    print(f"Prompt: {prompt}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Baseline top title: {response.baseline_top_title}")
    print(f"Inferred profile: {response.inferred_profile}")
    print("Retrieved sources:")
    for snippet in response.retrieved_context:
        print(f"  - {snippet.source} (score {snippet.score:.2f})")
    print("\nTop recommendations:")
    for rec in response.recommendations:
        print(f"  {rec.title} — {rec.artist}")
        print(f"    Genre/Mood: {rec.genre} / {rec.mood}")
        print(f"    Score: {rec.score:.2f}")
        print(f"    Because: {rec.explanation}")
    print("\nGrounded explanation:")
    print(response.grounded_explanation)
    if response.warnings:
        print("\nWarnings:")
        for warning in response.warnings:
            print(f"  - {warning}")
    print(f"\nLog file: {response.log_path}")


def main() -> None:
    advisor = build_default_advisor()
    for prompt in SHOWCASE_PROMPTS:
        _print_response(prompt, advisor)
    summary = run_evaluation(advisor)
    print_evaluation(summary)


if __name__ == "__main__":
    main()
