import csv
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
# Tunable weights (baseline). See model_card.md for a weight-shift experiment.
WEIGHT_GENRE = 2.0
WEIGHT_MOOD = 1.0
WEIGHT_ENERGY_SIMILARITY = 1.5
WEIGHT_ACOUSTIC_PREF = 0.5


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def _song_as_dict(song: Song) -> Dict[str, Any]:
    """Convert a Song dataclass row to the dict shape used by score_song."""
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _normalize_user_prefs(user_prefs: Dict) -> Dict[str, Any]:
    """Accept either CLI-style keys (genre, mood, energy) or UserProfile-style names."""
    genre = user_prefs.get("favorite_genre") or user_prefs.get("genre") or ""
    mood = user_prefs.get("favorite_mood") or user_prefs.get("mood") or ""
    energy = user_prefs.get("target_energy")
    if energy is None:
        energy = user_prefs.get("energy", 0.5)
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    return {
        "genre": str(genre).lower().strip(),
        "mood": str(mood).lower().strip(),
        "energy": float(energy),
        "likes_acoustic": likes_acoustic,
    }


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one catalog row against preferences; return total score and human-readable reasons."""
    prefs = _normalize_user_prefs(user_prefs)
    reasons: List[str] = []
    total = 0.0

    song_genre = str(song.get("genre", "")).lower().strip()
    song_mood = str(song.get("mood", "")).lower().strip()
    song_energy = float(song.get("energy", 0.0))
    song_acoustic = float(song.get("acousticness", 0.0))

    if prefs["genre"] and song_genre == prefs["genre"]:
        total += WEIGHT_GENRE
        reasons.append(f"genre match (+{WEIGHT_GENRE})")

    if prefs["mood"] and song_mood == prefs["mood"]:
        total += WEIGHT_MOOD
        reasons.append(f"mood match (+{WEIGHT_MOOD})")

    energy_gap = abs(song_energy - prefs["energy"])
    energy_sim = max(0.0, 1.0 - energy_gap)
    energy_points = WEIGHT_ENERGY_SIMILARITY * energy_sim
    total += energy_points
    reasons.append(f"energy similarity ({energy_sim:.2f}) (+{energy_points:.2f})")

    if prefs["likes_acoustic"]:
        acoustic_points = WEIGHT_ACOUSTIC_PREF * song_acoustic
        reasons.append(f"acoustic taste (+{acoustic_points:.2f})")
    else:
        acoustic_points = WEIGHT_ACOUSTIC_PREF * (1.0 - song_acoustic)
        reasons.append(f"non-acoustic lean (+{acoustic_points:.2f})")
    total += acoustic_points

    return total, reasons


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Rank songs by score for the given user and return the top k."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        scored: List[Tuple[float, Song]] = []
        for s in self.songs:
            sc, _ = score_song(prefs, _song_as_dict(s))
            scored.append((sc, s))
        scored.sort(key=lambda t: (-t[0], t[1].title.lower()))
        return [s for _, s in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a short explanation of how the song scores for this user."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        _, reasons = score_song(prefs, _song_as_dict(song))
        return "; ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            rows.append(row)
    return rows


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Score every song, rank by score descending, and return the top k with explanations."""
    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))
    scored.sort(
        key=lambda t: (-t[1], str(t[0].get("title", "")).lower()),
    )
    return scored[:k]
