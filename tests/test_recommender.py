from src.gemini_client import GeminiClient
from src.music_advisor import MusicAdvisor
from src.recommender import Song, UserProfile, Recommender, load_songs
from src.retrieval import MusicKnowledgeBase


class FakeGeminiClient:
    def __init__(self, response=None, should_fail=False):
        self.response = response or {
            "favorite_genre": "lofi",
            "favorite_mood": "focused",
            "target_energy": 0.35,
            "likes_acoustic": True,
            "request_summary": "The user wants focused, low-energy background music.",
            "confidence": 0.82,
            "uncertainty_notes": [],
            "retrieval_used": ["contexts.md", "mood_energy_guide.md"],
            "catalog_fit": "The catalog contains focused and chill low-energy songs.",
            "reasoning_trace": ["study implies focused mood", "background music implies lower energy"],
        }
        self.should_fail = should_fail

    def infer_preferences_with_gemini(self, prompt, retrieved_snippets, catalog_overview):
        if self.should_fail:
            raise RuntimeError("simulated Gemini failure")
        return self.response

    def explain_recommendation(self, prompt, retrieved_snippets, inferred_preferences, top_song, top_score, reasons):
        return f"{top_song['title']} fits because {reasons}. Sources: contexts.md."

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_retrieval_returns_context_for_study_prompt():
    kb = MusicKnowledgeBase("docs")
    snippets = kb.retrieve_context("I need something to study and focus", top_k=2)

    assert snippets
    assert any(item.source == "contexts.md" for item in snippets)


def test_gemini_json_parser_handles_fenced_json():
    client = GeminiClient.__new__(GeminiClient)
    parsed = client._parse_json_object(
        """```json
        {"favorite_genre": "pop", "target_energy": 0.8}
        ```"""
    )

    assert parsed["favorite_genre"] == "pop"
    assert parsed["target_energy"] == 0.8


def test_music_advisor_recommend_from_prompt_uses_gemini_profile():
    songs = load_songs("data/songs.csv")
    kb = MusicKnowledgeBase("docs")
    advisor = MusicAdvisor(songs=songs, knowledge_base=kb, llm_client=FakeGeminiClient(), log_path="logs/test_runs.jsonl")

    response = advisor.recommend_from_prompt("I need calm background music for studying and deep focus.", top_k=3)

    assert response.inferred_profile.favorite_genre == "lofi"
    assert response.recommendations
    assert response.recommendations[0].genre in {"lofi", "classical"}
    assert response.confidence > 0.5
    assert response.used_fallback is False


def test_music_advisor_falls_back_when_gemini_fails():
    songs = load_songs("data/songs.csv")
    kb = MusicKnowledgeBase("docs")
    advisor = MusicAdvisor(
        songs=songs,
        knowledge_base=kb,
        llm_client=FakeGeminiClient(should_fail=True),
        log_path="logs/test_runs.jsonl",
    )

    response = advisor.recommend_from_prompt("Recommend acoustic songs for studying.", top_k=3)

    assert response.used_fallback is True
    assert any("Gemini inference failed" in warning for warning in response.warnings)
    assert response.confidence < 0.6
