"""Local retrieval utilities for the music guidance corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Set, Tuple


TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class RetrievedSnippet:
    """A retrieved document snippet plus metadata used by the advisor."""

    source: str
    text: str
    score: float
    matched_terms: List[str]


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization that is stable across docs and prompts."""
    return TOKEN_RE.findall(text.lower())


class MusicKnowledgeBase:
    """Tiny local retriever over markdown guidance documents."""

    def __init__(self, docs_folder: str | Path):
        self.docs_folder = Path(docs_folder)
        self.documents = self.load_documents()
        self.index = self.build_index(self.documents)

    def load_documents(self) -> List[Tuple[str, str]]:
        """Load markdown or text files from the knowledge base folder."""
        documents: List[Tuple[str, str]] = []
        for path in sorted(self.docs_folder.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            documents.append((path.name, path.read_text(encoding="utf-8")))
        return documents

    def build_index(self, documents: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
        """Map token to the set of files containing that token."""
        index: Dict[str, Set[str]] = {}
        for filename, text in documents:
            for token in set(tokenize(text)):
                index.setdefault(token, set()).add(filename)
        return index

    def retrieve_context(self, query: str, top_k: int = 3) -> List[RetrievedSnippet]:
        """Return the highest scoring knowledge snippets for a user prompt."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidate_files: Set[str] = set()
        for token in query_tokens:
            candidate_files.update(self.index.get(token, set()))

        if not candidate_files:
            candidate_files = {filename for filename, _ in self.documents}

        scored: List[RetrievedSnippet] = []
        for filename, text in self.documents:
            if filename not in candidate_files:
                continue
            score, matched_terms = self._score_document(query_tokens, text)
            if score <= 0:
                continue
            scored.append(
                RetrievedSnippet(
                    source=filename,
                    text=self._make_snippet(text, matched_terms),
                    score=score,
                    matched_terms=matched_terms,
                )
            )

        scored.sort(key=lambda item: (-item.score, item.source))
        return scored[:top_k]

    def _score_document(self, query_tokens: List[str], text: str) -> Tuple[float, List[str]]:
        text_tokens = tokenize(text)
        text_token_set = set(text_tokens)
        matched = sorted({token for token in query_tokens if token in text_token_set})
        overlap_score = float(len(matched))

        lowered_text = text.lower()
        phrase_bonus = 0.0
        query_text = " ".join(query_tokens)
        if query_text and query_text in lowered_text:
            phrase_bonus += 2.0
        if "study" in query_tokens and "focused" in text_token_set:
            phrase_bonus += 1.0
        if "workout" in query_tokens and "high" in text_token_set and "energy" in text_token_set:
            phrase_bonus += 1.0
        if "acoustic" in query_tokens and "acousticness" in text_token_set:
            phrase_bonus += 1.0
        return overlap_score + phrase_bonus, matched

    def _make_snippet(self, text: str, matched_terms: List[str], limit: int = 500) -> str:
        if not matched_terms:
            return text[:limit].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in matched_terms):
                return line[:limit]
        return text[:limit].strip()
