"""
aggregator/scorer.py — Face-First Scoring v2
═════════════════════════════════════════════════════════════════════════
PHILOSOPHY CHANGE
─────────────────
Old scorer: combined_score = 0.4*name + 0.3*face + 0.3*sources
  Problem: 50M people share "John Smith" → name match = noise

New scorer:  combined_score = face_similarity * face_weight + metadata_bonus
  Face-verified match → minimum score 0.72 regardless of name match
  Name-only match     → maximum score 0.49 regardless of how many sources

SCORING BANDS
─────────────
  combined_score  verdict         meaning
  ─────────────   ──────────────  ─────────────────────────────────
  ≥ 0.85          CONFIRMED       face verified + profile consistent
  ≥ 0.70          CONFIRMED       face verified, limited metadata
  ≥ 0.55          POSSIBLE        face near-match OR strong metadata
  < 0.55          UNLIKELY        name match only, no face evidence

WEIGHTS
─────────────────────
  face_similarity  × 0.70   (primary — face IS the identity)
  name_match       × 0.10   (hint only)
  social_profile   × 0.08   (real social profile URL bonus)
  photo_available  × 0.05   (had downloadable photo at all)
  source_count     × 0.07   (found across multiple scrapers)
"""

from __future__ import annotations
import logging
from typing import Optional

from rapidfuzz import fuzz

import config

logger = logging.getLogger(__name__)

# Weights read from config so they can be tuned without touching this file
W_FACE    = config.W_FACE
W_NAME    = config.W_NAME
W_SOCIAL  = config.W_SOCIAL
W_PHOTO   = config.W_PHOTO
W_SOURCES = config.W_SOURCES

CONFIRMED_HIGH = config.VERDICT_CONFIRMED_HIGH
CONFIRMED_LOW  = config.VERDICT_CONFIRMED_LOW
POSSIBLE       = config.VERDICT_POSSIBLE

_SOCIAL_DOMAINS = {
    "linkedin.com","github.com","gitlab.com","twitter.com","x.com",
    "instagram.com","facebook.com","reddit.com","medium.com",
    "researchgate.net","orcid.org","behance.net","dribbble.com",
    "dev.to","stackoverflow.com","youtube.com","hackerrank.com",
    "kaggle.com","leetcode.com","npmjs.com","pypi.org",
}


def _is_social_url(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(d in host for d in _SOCIAL_DOMAINS)
    except Exception:
        return False


def _name_score(query_name: str, match_name: str) -> float:
    """Fuzzy name match using token_sort_ratio, capped at 0.8 — name alone cannot reach CONFIRMED."""
    if not query_name or not match_name:
        return 0.0
    ratio = fuzz.token_sort_ratio(query_name.lower(), match_name.lower())
    return round(min(ratio / 100.0, 0.8), 3)


def score_match(match: dict, query_name: str,
                query_location: str = "", query_company: str = "") -> dict:
    face_sim = float(match.get("face_similarity", 0.0))
    verified = bool(match.get("face_verified", False))

    face_score   = face_sim if verified else (face_sim * (0.3 if match.get("_verify_error") else 0.5))
    candidate_nm = (match.get("name") or match.get("title","").split(" - ")[0].split(" | ")[0]).strip()
    name_score   = _name_score(query_name, candidate_nm) if query_name else 0.0
    social_score = 1.0 if _is_social_url(match.get("url","")) else 0.0
    photo_score  = 1.0 if match.get("photo_url") else 0.0
    sources      = match.get("sources", [match.get("source","unknown")])
    # Weight source score by source_confidence if present (T3-3/T3-4)
    sc = match.get("source_confidence")
    if sc is not None:
        source_score = min(1.0, float(sc) * 1.5)
    else:
        source_score = min(1.0, len(set(sources)) / 3.0)

    combined = (W_FACE*face_score + W_NAME*name_score + W_SOCIAL*social_score
                + W_PHOTO*photo_score + W_SOURCES*source_score)

    # Location / company consistency micro-boost
    extra = 0.0
    if query_location and query_location.lower() in (match.get("location") or "").lower():
        extra += 0.03
    if query_company and query_company.lower() in (match.get("company") or match.get("bio") or "").lower():
        extra += 0.03
    combined = round(min(1.0, combined + extra), 4)

    if   verified and combined >= CONFIRMED_LOW:   verdict = "confirmed"
    elif combined >= POSSIBLE:                      verdict = "possible"
    else:                                           verdict = "unlikely"

    match["combined_score"]  = combined
    match["verdict"]         = verdict
    match["score_breakdown"] = {
        "face": round(W_FACE*face_score,3), "name": round(W_NAME*name_score,3),
        "social": round(W_SOCIAL*social_score,3), "photo": round(W_PHOTO*photo_score,3),
        "sources": round(W_SOURCES*source_score,3), "extra": round(extra,3),
        "raw_face_similarity": face_sim, "name_match": name_score,
        "face_verified": verified,
    }
    return match


def score_all(matches: list[dict], query_name: str,
              query_location: str = "", query_company: str = "",
              min_score: float = None) -> list[dict]:
    """Score all matches. Face-verified always rank above name-only.

    min_score overrides the default MIN_SCORE_KEEP cutoff. The preliminary
    (pre-face) pass passes a lower value so name/text candidates survive — the
    scorer weights face at 0.70, so text-only matches score low until the face
    engine confirms them in the final pass."""
    if not matches:
        return []
    thresh = config.MIN_SCORE_KEEP if min_score is None else min_score
    scored = [score_match(m, query_name, query_location, query_company) for m in matches]
    scored.sort(key=lambda m: (int(m.get("face_verified",False)), m.get("combined_score",0)),
                reverse=True)

    # Drop clearly noisy results below minimum threshold
    before_filter = len(scored)
    scored = [m for m in scored if m.get("combined_score", 0) >= thresh]
    filtered_out = before_filter - len(scored)
    if filtered_out:
        logger.info(f"Scorer v2: filtered {filtered_out} result(s) below threshold={thresh}")

    confirmed = sum(1 for m in scored if m["verdict"]=="confirmed")
    possible  = sum(1 for m in scored if m["verdict"]=="possible")
    logger.info(f"Scorer v2: {len(scored)} → {confirmed} confirmed, {possible} possible, "
                f"{len(scored)-confirmed-possible} unlikely")
    return scored