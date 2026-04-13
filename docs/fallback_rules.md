# Fallback And Reliability Rules

The song catalog is intentionally tiny, so the advisor must be honest about uncertainty.

- If the request conflicts with available songs, keep the closest supported genre or mood and explicitly mention the mismatch.
- When retrieval is weak, lower confidence instead of pretending the recommendation is precise.
- If Gemini fails or returns invalid JSON, use a local keyword fallback and mark the run as lower confidence.
- The final explanation should mention whether the answer was grounded in retrieved context or heuristic fallback logic.
