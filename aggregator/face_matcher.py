"""
aggregator/face_matcher.py
───────────────────────────
Downloads scraped profile photos and computes cosine similarity
against the query face embedding.

Resource-conscious design:
  - Downloads are sequential with short timeout (not parallel) to
    avoid hammering limited RAM with multiple concurrent image decodes
  - Skips URLs that fail quickly — no retries on photo downloads
  - Returns None scores gracefully so scorer handles missing face data

Threshold guide (Facenet512):
  >= 0.68  confirmed same person
  0.50-0.68 possible match
  < 0.35   different person
"""
import logging
import requests
import numpy as np
from typing import Optional
import cv2

import config
import embedding as emb_module

logger = logging.getLogger(__name__)


def score_all_results(
    all_results: dict,
    query_embedding: np.ndarray,
) -> dict:
    """
    Walk every match in all_results that has a photo URL.
    Download the photo, extract embedding, compute cosine similarity.
    Adds face_score field to each match in-place. Returns all_results.

    URL deduplication: each unique photo URL is downloaded exactly once.
    Scores are cached and re-used for duplicate URLs (e.g. 65 username
    matches sharing 5 Gravatar hashes → 5 downloads, not 65).
    """
    scored    = 0
    url_cache: dict[str, Optional[float]] = {}   # url → score (None = no face)

    for source, data in all_results.items():
        if not isinstance(data, dict):
            continue
        for match in data.get("matches", []):
            url = (
                match.get("photo_url")
                or match.get("avatar_url")
                or match.get("preview_url")
                or match.get("picture_url")
            )
            if not url:
                match["face_score"]      = None
                match["face_similarity"] = 0.0
                match["face_verified"]   = False
                continue

            if url not in url_cache:
                url_cache[url] = _score_url(url, query_embedding)

            score = url_cache[url]
            match["face_score"]      = score
            match["face_similarity"] = score if score is not None else 0.0
            match["face_verified"]   = (
                score is not None and score >= config.FACE_POSSIBLE
            )
            if score is not None:
                scored += 1
                logger.debug(
                    f"face_matcher: {source} '{match.get('username','?')}' "
                    f"score={score:.4f} ({emb_module.verdict(score)})"
                )

    logger.info(f"face_matcher: scored {scored} profile photo(s)")
    return all_results


def _score_url(url: str, query_embedding: np.ndarray) -> Optional[float]:
    try:
        r = requests.get(
            url,
            headers = config.BROWSER_HEADERS,
            timeout = 8,
            stream  = True,
        )
        r.raise_for_status()

        ct = r.headers.get("content-type", "")
        if "image" not in ct and "octet" not in ct:
            return None

        img_bytes = r.content
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # Skip tiny thumbnails / favicons (e.g. 32x32): their embeddings are
        # noise and produce spurious high similarity scores.
        if max(frame.shape[:2]) < getattr(config, "FACE_MATCH_MIN_PX", 120):
            return None

        result = emb_module.extract(frame)
        if result is None:
            return None

        return emb_module.cosine_similarity(query_embedding, result["embedding"])

    except Exception as e:
        logger.debug(f"face_matcher _score_url failed for {url}: {e}")
        return None