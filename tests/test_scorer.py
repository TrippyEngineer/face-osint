import config
from aggregator import scorer


def test_face_verified_outranks_name_only():
    face = {"face_similarity": 0.9, "face_verified": True,
            "name": "Jane Doe", "url": "https://github.com/jane"}
    name_only = {"face_similarity": 0.0, "face_verified": False,
                 "name": "Jane Doe", "url": "https://linkedin.com/in/jane"}
    out = scorer.score_all([name_only, face], query_name="Jane Doe")
    assert out, "face-verified match should survive the min-score filter"
    assert out[0].get("face_verified") is True
    assert face["verdict"] == "confirmed"
    assert face["combined_score"] >= config.VERDICT_CONFIRMED_LOW
    # genuinely outranks name-only on score (score_all mutates both dicts in
    # place before filtering, so this holds even if name_only is filtered out)
    assert name_only["verdict"] != "confirmed"
    assert face["combined_score"] > name_only["combined_score"]


def test_name_only_cannot_confirm():
    m = {"face_similarity": 0.0, "face_verified": False,
         "name": "John Smith", "url": "https://linkedin.com/in/jsmith"}
    scorer.score_match(m, "John Smith")
    assert m["verdict"] != "confirmed"
    assert m["combined_score"] < config.VERDICT_CONFIRMED_LOW


def test_name_score_capped_at_0_8():
    m = {"face_similarity": 0.0, "name": "Exact Name", "url": ""}
    scorer.score_match(m, "Exact Name")
    assert m["score_breakdown"]["name_match"] <= 0.8
