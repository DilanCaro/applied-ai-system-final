"""Streamlit interface for the Applied AI Music Advisor."""

from __future__ import annotations

import streamlit as st

from src.music_advisor import build_default_advisor


st.set_page_config(page_title="Applied AI Music Advisor", page_icon="🎵", layout="wide")
st.title("Applied AI Music Advisor")
st.caption("Gemini + retrieval + transparent scoring for grounded music recommendations.")

default_prompt = "I need calm background music for studying and deep focus."
prompt = st.text_area("Describe what you want to hear", value=default_prompt, height=120)

if st.button("Recommend songs", type="primary"):
    try:
        advisor = build_default_advisor()
        response = advisor.recommend_from_prompt(prompt, top_k=5)
    except Exception as exc:
        st.error(str(exc))
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Top Recommendations")
            for idx, rec in enumerate(response.recommendations, start=1):
                st.markdown(f"**{idx}. {rec.title} — {rec.artist}**")
                st.write(f"Genre: `{rec.genre}` | Mood: `{rec.mood}` | Score: `{rec.score:.2f}`")
                st.write(rec.explanation)

            st.subheader("Grounded Explanation")
            st.write(response.grounded_explanation)

        with right:
            st.subheader("Inferred Preferences")
            st.json(
                {
                    "favorite_genre": response.inferred_profile.favorite_genre,
                    "favorite_mood": response.inferred_profile.favorite_mood,
                    "target_energy": response.inferred_profile.target_energy,
                    "likes_acoustic": response.inferred_profile.likes_acoustic,
                    "summary": response.inferred_profile.request_summary,
                    "model_confidence": response.inferred_profile.model_confidence,
                    "pipeline_confidence": response.confidence,
                }
            )

            st.subheader("Retrieved Sources")
            for snippet in response.retrieved_context:
                st.markdown(f"**{snippet.source}**")
                st.write(snippet.text)
                st.caption(f"Matched terms: {', '.join(snippet.matched_terms)} | Score: {snippet.score:.2f}")

            if response.warnings:
                st.subheader("Warnings")
                for warning in response.warnings:
                    st.warning(warning)

            st.subheader("Baseline Comparison")
            st.write(f"Baseline top title: `{response.baseline_top_title}`")
            st.json(response.baseline_preferences)
