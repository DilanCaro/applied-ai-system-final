# 🎧 Model Card: Applied AI Music Advisor

## 1. Model Name

**Applied AI Music Advisor**

---

## 2. Intended Use

This system suggests **up to five songs** from a small CSV catalog for a **single user** at a time. The final version accepts a free-text request, retrieves local music-guidance documents, asks Gemini to infer a structured listener profile, and then ranks songs using transparent scoring rules. It is meant for **learning and demos**, not for a real product, streaming service, or any high-stakes decision.

---

## 3. How the Model Works

### Explain your scoring approach in simple language.  

The system now has two layers:

1. A **retrieval + Gemini inference layer** that converts a natural-language request into structured fields such as genre, mood, target energy, and acoustic preference.
2. A **transparent ranking layer** that scores each song using those fields.

The ranking model still adds points for genre match, mood match, energy similarity, and acoustic alignment. The important extension is that Gemini now chooses those ranking inputs using retrieved local guidance rather than a user filling out fields manually.

---

## 4. Data

The catalog has **18** hand-authored tracks in `data/songs.csv`. Genres include pop, lofi, rock, jazz, ambient, synthwave, indie pop, metal, country, classical, reggae, hip hop, folk, and punk. Moods include happy, chill, intense, relaxed, focused, moody, and sad.

The RAG layer also uses a local document set in `docs/` with guidance on:

- listening contexts
- mood and energy interpretation
- genre aliases
- fallback and uncertainty rules

---

## 5. Strengths

The design is still **inspectable**: the ranking weights are explicit, retrieved sources are shown, Gemini output is constrained to JSON, and each run is logged. The same end-to-end advisor powers the CLI, Streamlit app, and evaluation harness, which makes it easier to spot mismatches between the demo and the tested path.

---

## 6. Limitations and Bias

The catalog is still tiny, so even grounded inference can only recommend from limited options. Gemini can also misread a prompt or return malformed JSON, which is why the system includes fallback parsing, warnings, and confidence scoring. Exact catalog labels still matter, even with alias guidance, so some subtle genre requests will collapse into the closest supported label.

---

## 7. Evaluation

Testing now covers retrieval, Gemini JSON parsing, fallback behavior, and end-to-end recommendation flow. The evaluation harness runs prompts for study/focus, workout, chill acoustic listening, conflicting emotional requests, and vague unsupported requests.

The main quantitative signals are:

- retrieval hit rate
- recommendation alignment rate
- average confidence
- count of low-confidence cases

The most interesting behavior appears on conflicting prompts like **sad but energetic pop**, where the system can return a plausible match while still surfacing uncertainty.

---

## 8. Future Work

- Add richer retrieval documents or user-authored playlists as a second knowledge source.
- Use `valence` and `tempo_bpm` more directly in ranking.
- Add diversity constraints so one artist cannot dominate the top five.
- Add human evaluation rubrics alongside automated checks.

---

## 9. Personal Reflection

Building this version showed me that retrieval and prompting are only useful when they feed a system that can still be audited. Gemini made the input experience much more natural, but the project only became trustworthy after I added structured outputs, fallback handling, and explicit reliability signals. If I kept extending it, I would prioritize larger datasets, more human evaluation, and stronger safeguards against overconfident recommendations.
