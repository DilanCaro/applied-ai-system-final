# Reflection Notes

## What this project taught me

The biggest lesson from this final version was that a polished AI feature needs both intelligence and evidence. A natural-language music prompt feels much more useful than a fixed preference form, but it only became trustworthy once I required Gemini to use retrieved local guidance and return structured JSON that the ranking engine could verify.

## Reliability surprises

The most surprising behavior showed up on conflicting prompts like `sad but energetic pop`. Gemini could map the request into a reasonable structured profile, but the tiny catalog still limited how exact the final recommendation could be. That made confidence scoring and warnings much more important than I expected.

## Limitations and bias

- The catalog is tiny and hand-curated, so some genres and emotional combinations are underrepresented.
- The system can over-prefer whatever labels happen to exist in the data instead of what a real user might mean.
- Retrieval guidance helps normalize prompts, but it cannot create catalog coverage that does not exist.

## Misuse and mitigation

This system could be misused if someone treated its outputs as highly personalized or authoritative. I reduce that risk by showing retrieved sources, logging each run, surfacing warnings, and lowering confidence when requests are vague, conflicting, or unsupported.

## Collaboration with AI

One helpful AI contribution during development was suggesting how to structure the pipeline as retrieval -> Gemini inference -> ranking -> reliability signals, which made the system much more coherent than a loose chatbot add-on. One flawed suggestion I had to avoid was letting the model produce free-form recommendation text without constraining it to catalog labels; that would have sounded smart while disconnecting the output from the actual scoring logic.
