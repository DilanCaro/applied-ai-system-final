# Applied AI Music Advisor

## Original Project

This project extends my earlier **Music Recommender Simulation** from Module 3. The original version ranked songs from a CSV using explicit preferences like genre, mood, energy, and acousticness, then returned transparent rule-based explanations for the top matches.

For the final project, I turned that prototype into a fuller applied AI system: users can now enter a natural-language request, the system retrieves local music-guidance documents, Gemini infers a structured listener profile from those sources, and the ranking engine returns grounded recommendations with confidence and logging.

## Title And Summary

Applied AI Music Advisor is a **Gemini + RAG music recommender** built with Python. It combines:

- local retrieval over a small music knowledge base
- Gemini-based intent inference using the Google AI Studio API key
- a transparent ranking engine over a hand-authored song catalog
- confidence scoring, warnings, and JSONL logs for reliability review

The goal is not to imitate Spotify-scale personalization. It is to demonstrate a trustworthy, explainable AI pipeline where retrieval actually changes recommendation behavior and where failures are visible instead of hidden.

## Architecture Overview

The architecture source file is in [assets/system_architecture.md](assets/system_architecture.md).

High-level flow:

1. The user enters a free-text request.
2. The retriever searches local guidance documents in `docs/`.
3. Retrieved snippets are sent to Gemini along with allowed catalog labels.
4. Gemini returns a structured listener profile in JSON.
5. The ranking engine scores songs with those inferred preferences.
6. The app reports recommendations, retrieved sources, confidence, and warnings.
7. Each run is logged to `logs/music_advisor_runs.jsonl`.

## Setup Instructions

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended for the Google Gemini dependencies.

3. Set your Google AI Studio API key.

```bash
export GOOGLE_API_KEY="your_key_here"
```

You can also place the key in a local `.env` file:

```bash
GOOGLE_API_KEY=your_key_here
```

4. Run the CLI demo.

```bash
python3 -m src.main
```

5. Or run the Streamlit app.

```bash
python3 -m streamlit run streamlit_app.py
```

6. Run the tests.

```bash
python3 -m pytest -q
```

## Sample Interactions

### Example 1

Input:

```text
I need calm background music for studying and deep focus.
```

Expected behavior:

- retrieval should surface `contexts.md`
- Gemini should infer lower energy and a focus-oriented mood
- top recommendations should skew toward `lofi` or other low-energy, concentration-friendly songs

### Example 2

Input:

```text
Give me something intense and high energy for a workout.
```

Expected behavior:

- retrieval should surface workout guidance
- Gemini should infer high energy and likely `intense`
- top recommendations should favor tracks like rock, metal, or intense pop entries

### Example 3

Input:

```text
Recommend sad but energetic pop for a dramatic run.
```

Expected behavior:

- the system should preserve the conflict instead of flattening it
- confidence may drop if the catalog has limited exact matches
- warnings should mention any mismatch between the request and available songs

## Design Decisions

- I kept the original rule-based ranking engine because it is inspectable and easy to test.
- I added RAG as the new AI feature so the system can interpret free-text requests using retrieved local guidance instead of hard-coded form fields alone.
- Gemini is used for structured inference rather than unconstrained generation, which makes the pipeline easier to validate and log.
- I kept both a CLI and Streamlit interface so the same core logic supports reproducible testing and a portfolio-ready demo.
- I added a heuristic fallback path so API failures are observable and recoverable rather than crashing the whole system.

## Reliability And Testing

This project includes several reliability signals:

- automated tests in `tests/test_recommender.py`
- JSON parsing validation for Gemini output
- retrieval hit checks for benchmark prompts
- pipeline confidence scoring based on retrieval strength, profile completeness, top-result alignment, and fallback use
- warning messages when retrieval is weak or Gemini output fails
- JSONL logging for every run

The evaluation harness in `src/evaluation.py` runs benchmark prompts and prints:

- retrieval hit rate
- alignment rate
- average confidence
- number of low-confidence cases

Testing summary template after running locally:

```text
X/X tests passed. Retrieval hit rate was Y. Average confidence was Z. Low-confidence cases mostly occurred when prompts conflicted with the tiny catalog or Gemini fallback was triggered.
```

## Reflection

This project taught me that “AI integration” is only meaningful when the model changes system behavior. The most useful design choice here was forcing Gemini to return structured ranking inputs instead of letting it generate vague recommendation prose. The biggest challenge was reliability: a small catalog makes it easy for the system to sound confident when it should actually admit a partial match.

Working with AI during development was helpful for brainstorming failure cases and scaffolding code, but I still had to verify grounding, JSON parsing, and whether retrieval actually affected the rankings. One flawed AI-style shortcut would have been adding a standalone chatbot explanation without wiring it into the ranking logic; I avoided that by making retrieved context and Gemini inference feed the scorer directly.

## Ethics And Limitations

- The catalog is tiny and hand-authored, so it does not represent real listener diversity.
- Genre and mood labels are simplified and may flatten nuanced music preferences.
- The system could be misused by overstating its confidence; this is why it exposes warnings, confidence, and retrieved evidence.
- Bias can appear through catalog coverage: if few songs represent a mood or genre, the system may over-recommend the closest mainstream alternative.

## Portfolio And Presentation

- GitHub repo: add your final public repository link here
- Loom walkthrough: add your final Loom link here
- Portfolio reflection: this project shows that I can turn a small prototype into an end-to-end AI system with retrieval, model integration, testing, and transparent reliability signals

## Supporting Files

- [model_card.md](model_card.md)
- [reflection.md](reflection.md)
- [assets/system_architecture.md](assets/system_architecture.md)

## Screenshots

![Lo-fi](img/lofi.png)

![high energy](img/high_energy.png)

![Deep intense Rock](img/deep_intense_Rock.png)

![Edge Case](img/edge_case.png)
