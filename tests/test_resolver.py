from aggregator import resolver


def test_resolve_empty_returns_no_results():
    out = resolver.resolve("Nobody", [])
    assert out["verdict"] == "no_results"
    assert out["combined_score"] == 0.0


def test_resolve_merges_same_email():
    a = {"name": "Jane", "email": "jane@x.com", "combined_score": 0.8,
         "verdict": "confirmed", "url": "https://github.com/jane", "source": "github"}
    b = {"name": "Jane D", "email": "jane@x.com", "combined_score": 0.6,
         "verdict": "possible", "url": "https://twitter.com/jane", "source": "twitter"}
    out = resolver.resolve("Jane", [a, b])
    assert out["email"] == "jane@x.com"
    assert len(out["all_profiles"]) == 2
    assert out["combined_score"] == 0.8


def test_resolve_picks_highest_scoring_cluster():
    c1 = {"name": "Alice", "combined_score": 0.9, "url": "https://github.com/alice",
          "source": "github", "email": "alice@a.com"}
    c2 = {"name": "Bob", "combined_score": 0.3, "url": "https://reddit.com/u/bob",
          "source": "reddit", "email": "bob@b.com"}
    out = resolver.resolve("Alice", [c1, c2])
    assert out["combined_score"] == 0.9
    assert out["resolved_name"] in ("Alice", "alice")
