"""
embedding.py
─────────────
DeepFace-based face detection and Facenet512 embedding extraction.

Uses DeepFace because:
  • Already proven working on this machine (attendance.py)
  • Zero extra install on Windows — same pip packages
  • Facenet512 gives 512D L2-normalised embeddings (identical to ArcFace quality)
  • Consistent with the rest of the project

Cosine similarity for Facenet512:
    >= 0.68  confirmed same person
    0.50-0.68 possible match
    < 0.35   different person

Thread safety:
    extract() and compare() are stateless — safe to call from any thread.
    DeepFace loads the model on first call and caches it internally.
"""

import os
import logging
import numpy as np
import cv2
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Tell DeepFace where to cache models — keeps them in our project folder
os.environ["DEEPFACE_HOME"]              = str(config.MODELS_DIR)
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"


def extract(frame: np.ndarray) -> Optional[dict]:
    """
    Detect the largest face in *frame* and return its Facenet512 embedding.

    Args:
        frame: BGR uint8 numpy array (from OpenCV)

    Returns dict with:
        embedding   — 512D float32 L2-normalised numpy array
        face_crop   — 160x160 BGR face crop (for saving to folder)
        bbox        — (x, y, w, h) detection box in frame coordinates
        confidence  — detection confidence 0–1

    Returns None if no face is detected.
    """
    try:
        from deepface import DeepFace

        results = DeepFace.represent(
            img_path        = frame,
            model_name      = config.DEEPFACE_MODEL,
            detector_backend= config.DEEPFACE_DETECTOR,
            enforce_detection = config.DEEPFACE_ENFORCE,
            align           = True,
        )

        if not results:
            return None

        # Pick the detection with highest confidence
        best = max(results, key=lambda r: r.get("face_confidence", 0))

        if best.get("face_confidence", 0) < 0.5:
            logger.debug(f"Low confidence detection: {best.get('face_confidence', 0):.3f}")
            return None

        emb  = np.array(best["embedding"], dtype=np.float32)
        emb  = _l2_normalise(emb)

        # Extract face crop using facial_area bounding box
        fa   = best.get("facial_area", {})
        crop = _crop_face(frame, fa)

        return {
            "embedding":  emb,
            "face_crop":  crop,
            "bbox":       (fa.get("x", 0), fa.get("y", 0),
                           fa.get("w", 0), fa.get("h", 0)),
            "confidence": round(best.get("face_confidence", 0), 4),
        }

    except Exception as e:
        logger.debug(f"embedding.extract() failed: {e}")
        return None


def extract_from_url(url: str) -> Optional[np.ndarray]:
    """
    Download an image from *url* and extract its embedding.
    Used by face_matcher to compare scraped profile photos.

    Returns L2-normalised 512D embedding or None.
    """
    try:
        import requests
        from io import BytesIO
        from PIL import Image

        r = requests.get(
            url,
            headers = config.BROWSER_HEADERS,
            timeout = config.HTTP_TIMEOUT_S,
            stream  = True,
        )
        r.raise_for_status()

        ct = r.headers.get("content-type", "")
        if "image" not in ct and "octet" not in ct:
            return None

        img   = Image.open(BytesIO(r.content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        result = extract(frame)
        return result["embedding"] if result else None

    except Exception as e:
        logger.debug(f"extract_from_url failed for {url}: {e}")
        return None


def prewarm() -> None:
    """
    Load the Facenet512 model into memory ahead of the first search.

    PERF: the first extract() call pays a ~30-40s cold-start (TensorFlow init +
    Facenet512 weight load). Calling this once in a background thread at app
    startup moves that cost off the user's first search. DeepFace caches the
    model internally, so the subsequent real extract() is fast. Safe no-op on
    failure — extract() still loads lazily.
    """
    try:
        from deepface import DeepFace
        DeepFace.build_model(config.DEEPFACE_MODEL)
        logger.info(f"DeepFace model pre-warmed: {config.DEEPFACE_MODEL}")
    except Exception as e:
        logger.warning(f"DeepFace pre-warm failed (will load lazily on first search): {e}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised embeddings.
    Since both are unit vectors: similarity = dot product.
    Range: -1 to 1. Same person typically > 0.68 for Facenet512.
    """
    a = _l2_normalise(a)
    b = _l2_normalise(b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def verdict(score: Optional[float]) -> str:
    """Human-readable verdict from a similarity score."""
    if score is None:
        return "unknown"
    if score >= config.FACE_CONFIRMED:
        return "confirmed"
    if score >= config.FACE_POSSIBLE:
        return "possible"
    return "different"


# ── Internals ─────────────────────────────────────────────────────────────
def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _crop_face(frame: np.ndarray, fa: dict, padding: float = 0.20) -> np.ndarray:
    """Crop the face region with padding, resize to 160x160."""
    h, w   = frame.shape[:2]
    x      = int(fa.get("x", 0))
    y      = int(fa.get("y", 0))
    fw     = int(fa.get("w", w))
    fh     = int(fa.get("h", h))

    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x1    = max(0, x - pad_x)
    y1    = max(0, y - pad_y)
    x2    = min(w, x + fw + pad_x)
    y2    = min(h, y + fh + pad_y)

    crop  = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame, (160, 160))
    return cv2.resize(crop, (160, 160))


def hires_face_crop(
    frame: np.ndarray,
    bbox,
    padding:  float = 0.4,
    max_side: int   = 1024,
    min_side: int   = 400,
) -> np.ndarray:
    """High-resolution face crop for REVERSE-IMAGE SEARCH (Lens / Bing / SerpApi).

    Unlike `_crop_face` (which downsizes to the 160x160 Facenet input), this
    keeps native resolution so the reverse engines have real detail to match on.
    A 160px thumbnail returns garbage; this returns the face at full quality.

    - Generous padding keeps hair/jaw/shoulders (helps Lens recall).
    - Long side capped at `max_side` so uploads stay fast.
    - Never upscales — only downsizes when above the cap.
    - If there is no usable bbox, or the padded crop is smaller than `min_side`,
      the full frame is returned (more pixels/context than a tiny crop).

    bbox: (x, y, w, h) in frame coordinates, as returned by extract().
    """
    try:
        H, W = frame.shape[:2]
        x, y, w, h = (int(v) for v in (bbox or (0, 0, 0, 0)))
        crop = None
        if w > 0 and h > 0:
            px, py = int(w * padding), int(h * padding)
            x1, y1 = max(0, x - px), max(0, y - py)
            x2, y2 = min(W, x + w + px), min(H, y + h + py)
            c = frame[y1:y2, x1:x2]
            if c.size and max(c.shape[:2]) >= min_side:
                crop = c
        if crop is None:
            crop = frame                       # no face / crop too small → full frame
        long_side = max(crop.shape[:2])
        if long_side > max_side:               # downscale only, never upscale
            s = max_side / long_side
            crop = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                interpolation=cv2.INTER_AREA,
            )
        return crop
    except Exception as e:
        logger.debug(f"hires_face_crop failed: {e}")
        return frame
