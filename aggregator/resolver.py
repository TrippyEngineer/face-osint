"""
aggregator/resolver.py
───────────────────────
Entity resolution — groups scraped profiles into identity clusters
using Union-Find and produces one clean resolved identity dict.

Edge weights between two profiles:
  Same email (exact)         → 0.90
  Username in other URL      → 0.85
  Face scores both confirmed → 1.00
  Same employer (fuzzy ≥75%) → 0.45
  Same location (fuzzy ≥70%) → 0.30
  Name similarity (≥80%)     → 0.40

Connected components with edge ≥ 0.30 are merged.
The best cluster (highest avg combined_score) is the resolved identity.
"""
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz
    def _ratio(a, b): return _fuzz.ratio(a, b) / 100.0
except ImportError:
    try:
        from Levenshtein import ratio as _lev
        def _ratio(a, b): return _lev(a, b)
    except ImportError:
        def _ratio(a, b):
            a, b = a.lower(), b.lower()
            if a == b: return 1.0
            if not a or not b: return 0.0
            return sum(c1==c2 for c1,c2 in zip(a,b)) / max(len(a),len(b))


EDGE_THRESHOLD = 0.30


def resolve(query_name: str, scored_matches: list) -> dict:
    """
    Takes scored_matches from scorer.score_all().
    Returns one resolved identity dict with all evidence merged.
    """
    if not scored_matches:
        return _empty(query_name)

    # Build Union-Find
    parent = list(range(len(scored_matches)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(len(scored_matches)):
        for j in range(i + 1, len(scored_matches)):
            w = _edge_weight(scored_matches[i], scored_matches[j])
            if w >= EDGE_THRESHOLD:
                union(i, j)

    # Group into clusters
    clusters: dict = {}
    for i in range(len(scored_matches)):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Pick the cluster with the highest average combined_score
    def cluster_score(indices):
        return sum(scored_matches[i].get("combined_score", 0) for i in indices) / len(indices)

    best_indices = max(clusters.values(), key=cluster_score)
    best_profiles = [scored_matches[i] for i in best_indices]

    best_profiles = _dedup_profiles(best_profiles)
    identity = _merge(query_name, best_profiles)
    logger.info(
        f"resolver: {len(best_profiles)} profile(s) merged → "
        f"verdict={identity['verdict']} score={identity['combined_score']:.3f}"
    )
    return identity


def _normalize_url(url: str) -> str:
    """Strip scheme, www., and trailing slash for deduplication comparison."""
    try:
        url = url.lower().strip()
        url = url.replace("https://", "").replace("http://", "")
        url = url.lstrip("www.")
        url = url.rstrip("/")
    except Exception:
        pass
    return url


def _dedup_profiles(profiles: list) -> list:
    """
    Remove redundant URL variants and merge same-username same-platform
    duplicates. Keeps the higher-scored profile. O(n²) — lists are small.
    """
    if len(profiles) <= 1:
        return profiles

    keep = [True] * len(profiles)

    for i in range(len(profiles)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(profiles)):
            if not keep[j]:
                continue

            # Compare normalised URLs
            url_i = _normalize_url(profiles[i].get("url") or profiles[i].get("profile_url") or "")
            url_j = _normalize_url(profiles[j].get("url") or profiles[j].get("profile_url") or "")
            if url_i and url_i == url_j:
                # Keep higher-scored one
                score_i = profiles[i].get("combined_score", 0)
                score_j = profiles[j].get("combined_score", 0)
                if score_j > score_i:
                    keep[i] = False
                else:
                    keep[j] = False
                continue

            # Same username on same platform → merge
            uname_i = (profiles[i].get("username") or "").lower().strip()
            uname_j = (profiles[j].get("username") or "").lower().strip()
            plat_i  = (profiles[i].get("platform") or profiles[i].get("source") or "").lower()
            plat_j  = (profiles[j].get("platform") or profiles[j].get("source") or "").lower()
            if uname_i and uname_i == uname_j and plat_i and plat_i == plat_j:
                score_i = profiles[i].get("combined_score", 0)
                score_j = profiles[j].get("combined_score", 0)
                if score_j > score_i:
                    keep[i] = False
                else:
                    keep[j] = False

    result = [p for p, k in zip(profiles, keep) if k]
    removed = len(profiles) - len(result)
    if removed:
        logger.debug(f"_dedup_profiles: removed {removed} redundant profile(s)")
    return result


def _edge_weight(a: dict, b: dict) -> float:
    w = 0.0

    fa, fb = a.get("face_score"), b.get("face_score")

    # Face VETO: if both have a face score and they actively contradict (one is
    # a match, the other clearly rejected) they are NOT the same person — never
    # merge, no matter how well name/email/employer line up. Stops same-name
    # strangers (e.g. a namesake whose face scored 0.19) from being fused into a
    # face-confirmed identity and leaking their contact details into it.
    if (fa is not None and fb is not None
            and max(fa, fb) >= config.FACE_POSSIBLE
            and min(fa, fb) < config.FACE_REJECTED):
        return 0.0

    # Face score agreement (only when BOTH are at least a possible match)
    if (fa is not None and fb is not None
            and min(fa, fb) >= config.FACE_POSSIBLE):
        avg = (fa + fb) / 2.0
        if avg >= config.FACE_CONFIRMED:
            w = max(w, 1.00)
        else:
            w = max(w, 0.60)

    # Email exact match
    ea = (a.get("email") or "").lower().strip()
    eb = (b.get("email") or "").lower().strip()
    if ea and ea == eb:
        w = max(w, 0.90)

    # Username cross-reference in URLs
    ua   = (a.get("username") or "").lower()
    ub   = (b.get("username") or "").lower()
    urla = (a.get("url") or a.get("profile_url") or "").lower()
    urlb = (b.get("url") or b.get("profile_url") or "").lower()
    if ua and ua in urlb:
        w = max(w, 0.85)
    if ub and ub in urla:
        w = max(w, 0.85)

    # Employer fuzzy match
    empa = (a.get("company") or a.get("affiliation") or "").lower()
    empb = (b.get("company") or b.get("affiliation") or "").lower()
    if empa and empb and _ratio(empa, empb) >= 0.75:
        w = max(w, 0.45)

    # Location fuzzy match
    loca = (a.get("location") or "").lower()
    locb = (b.get("location") or "").lower()
    if loca and locb and _ratio(loca, locb) >= 0.70:
        w = max(w, 0.30)

    # Name similarity
    na = (a.get("name") or a.get("username") or "").lower()
    nb = (b.get("name") or b.get("username") or "").lower()
    if na and nb and _ratio(na, nb) >= 0.80:
        w = max(w, 0.40)

    return w


def _clean_name(p: dict) -> str:
    """Strip title suffixes like ' - Programme officer' / ' | LinkedIn'."""
    nm = (p.get("name") or p.get("username") or "").strip()
    for sep in (" - ", " | ", " – ", " — "):
        nm = nm.split(sep)[0]
    return nm.strip()


def _merge(query_name: str, profiles: list) -> dict:
    # Identity is FACE-anchored: only face-verified profiles define who this is.
    verified = [p for p in profiles
                if p.get("face_verified") or (p.get("face_score") or 0) >= config.FACE_POSSIBLE]
    # Among verified, prefer those whose name actually matches the query — drops
    # face-similar connections / look-alikes (e.g. "Pujya Ghosh") from driving
    # the identity's name + contact details.
    qn = (query_name or "").lower().strip()
    name_consistent = [p for p in verified
                       if qn and _ratio(qn, _clean_name(p).lower()) >= 0.6]
    core = name_consistent or verified or profiles   # graceful fallback (name-only clusters)

    best = max(core, key=lambda p: (p.get("face_score") or 0.0, p.get("combined_score", 0)))

    urls    = list({p.get("url") or p.get("profile_url") or "" for p in profiles if p.get("url") or p.get("profile_url")})
    sources = list({p.get("source", "") for p in profiles if p.get("source")})
    faces   = ([p["face_score"] for p in verified if p.get("face_score") is not None]
               or [p["face_score"] for p in profiles if p.get("face_score") is not None])

    # A face match alone (especially reverse-image look-alikes) is a *candidate*,
    # not proof. Only call it CONFIRMED when the query NAME corroborates the face;
    # for a photo-only search or a name mismatch, cap the verdict at POSSIBLE.
    verdict = best.get("verdict", "possible")
    if verdict == "confirmed" and not name_consistent:
        verdict = "possible"

    return {
        "query_name":     query_name,
        "resolved_name":  _clean_name(best) or query_name,
        "face_score":     max(faces) if faces else None,
        "combined_score": max((p.get("combined_score", 0) for p in core), default=0.0),
        "verdict":        verdict,
        "sources":        sources,
        "profile_urls":   [u for u in urls if u],
        # contact/identity fields come ONLY from the core (face-verified,
        # query-consistent) profiles — never from same-name strangers.
        "email":          _first(core, "email"),
        "company":        _first(core, "company") or _first(core, "affiliation"),
        "location":       _first(core, "location"),
        "username":       _first(core, "username"),
        "bio":            _first(core, "bio"),
        "photo_url":      _first(core, "photo_url") or _first(core, "avatar_url"),
        "all_profiles":   profiles,
    }


def _empty(name: str) -> dict:
    return {
        "query_name": name, "resolved_name": name,
        "face_score": None, "combined_score": 0.0,
        "verdict": "no_results", "sources": [],
        "profile_urls": [], "all_profiles": [],
    }


def _first(profiles: list, field: str):
    for p in profiles:
        v = p.get(field)
        if v:
            return v
    return None
