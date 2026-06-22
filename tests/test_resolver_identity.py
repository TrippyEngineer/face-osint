"""Resolver identity tests — face-anchored, no same-name/connection contamination.
Encodes the real 'Neeraj Jain' failure: a face-rejected namesake (GitHub dev) was
fused in and its email/company stamped onto a face-matched person, and a
face-similar connection (Pujya Ghosh) drove the identity name."""
from aggregator import resolver


def test_edge_weight_face_veto_blocks_same_name_different_person():
    # same name, but faces actively contradict (match vs rejected) -> never merge
    a = {"name": "Neeraj Jain", "face_score": 0.95}
    b = {"name": "Neeraj Jain", "face_score": 0.19}
    assert resolver._edge_weight(a, b) == 0.0


def test_edge_weight_merges_two_face_confirmed():
    a = {"name": "Neeraj Jain", "face_score": 0.95}
    b = {"name": "Neeraj Jain", "face_score": 0.93}
    assert resolver._edge_weight(a, b) >= 1.0


def test_edge_weight_name_only_still_merges():
    # no face evidence on either side -> name match may still cluster them
    a = {"name": "Neeraj Jain"}
    b = {"name": "Neeraj Jain"}
    assert resolver._edge_weight(a, b) >= 0.40


def test_resolved_identity_is_face_anchored_and_uncontaminated():
    scored = [
        {"name": "Neeraj Jain - Healthcare Advocate", "face_score": 0.95,
         "face_verified": True, "combined_score": 0.82, "verdict": "confirmed",
         "source": "serpapi_lens", "url": "https://ihwcouncil.org/neeraj-jain/"},
        {"name": "PUJYA GHOSH - Programme officer", "face_score": 0.94,
         "face_verified": True, "combined_score": 0.83, "verdict": "confirmed",
         "source": "serpapi_lens", "url": "https://linkedin.com/in/pujya-ghosh"},
        {"name": "Neeraj Jain", "username": "CodeWithNJ", "face_score": 0.19,
         "face_verified": False, "combined_score": 0.30, "verdict": "unlikely",
         "email": "neerajj9@gmail.com", "company": "myoperator",
         "location": "Gurgaon", "source": "github", "url": "https://github.com/CodeWithNJ"},
    ]
    ident = resolver.resolve("Neeraj Jain", scored)
    # the WRONG Neeraj's (face-rejected namesake) contact info must NOT leak in
    assert ident.get("email") != "neerajj9@gmail.com"
    assert (ident.get("company") or "") != "myoperator"
    # identity name = the real face-matched, query-consistent person, not the connection
    rn = (ident.get("resolved_name") or "").lower()
    assert "neeraj" in rn and "pujya" not in rn
    # the face-rejected namesake must not be merged into the resolved cluster
    assert "github.com/CodeWithNJ" not in " ".join(ident.get("profile_urls", []))


def test_photo_only_face_match_is_capped_to_possible():
    # no query name: a single face-verified look-alike must NOT be CONFIRMED
    scored = [{"name": "Some Lookalike", "face_score": 0.94, "face_verified": True,
               "combined_score": 0.81, "verdict": "confirmed",
               "source": "serpapi_lens", "url": "https://example/p"}]
    ident = resolver.resolve("", scored)
    assert ident["verdict"] != "confirmed"   # capped to possible without name corroboration


def test_name_corroborated_face_match_confirms():
    scored = [{"name": "Neeraj Jain", "face_score": 0.94, "face_verified": True,
               "combined_score": 0.85, "verdict": "confirmed",
               "source": "serpapi_lens", "url": "https://example/n"}]
    ident = resolver.resolve("Neeraj Jain", scored)
    assert ident["verdict"] == "confirmed"   # name corroborates the face
