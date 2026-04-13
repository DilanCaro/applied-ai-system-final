# Genre Alias Guide

Use only the genres available in the catalog when normalizing a request.

- `indie pop` should stay `indie pop`; if a user says plain `pop`, related indie-pop tracks may still be reasonable alternatives.
- `lo-fi` and `study beats` should normalize to `lofi`.
- `rap` should normalize to `hip hop`.
- `heavy` or `aggressive guitar` requests often fit `rock`, `metal`, or `punk`; prefer the closest supported label mentioned by the user.
- If the request is vague and no genre is explicit, use context and mood first instead of inventing a new genre.
