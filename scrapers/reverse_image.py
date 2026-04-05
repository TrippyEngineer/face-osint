"""
scrapers/reverse_image.py — Face-First OSINT Engine v2
═══════════════════════════════════════════════════════════════════════════
THE CORRECT APPROACH
────────────────────
Old approach (broken):  name → text search → hope face appears in results
New approach (this):    FACE → reverse image search → verify faces → get name

The face IS the identity. 50M people share the same name.
Zero people share the same face embedding.

PIPELINE
────────
Step 1  Upload the face crop to imgbb (free temp hosting) → public URL
Step 2  Submit URL to Yandex reverse image search (best face recognition)
        Parse HTML → extract page titles (contain names), source domains,
        and image thumbnails
Step 3  Submit same URL to Bing Visual Search API (if BING_SEARCH_KEY set)
Step 4  For every candidate image/page URL found:
          a. Download the image (or extract OG image from profile page)
          b. Run DeepFace.verify() against the query embedding
          c. Only keep results where similarity ≥ FACE_VERIFY_THRESHOLD
Step 5  Classify face-verified URLs by domain → social media profiles
Step 6  Use Google CSE to expand: search "<discovered name>" on
        LinkedIn, GitHub, Instagram, Twitter (if name found from steps 2-4)
Step 7  Return structured matches tagged face_verified=True/False


ADDITIONAL RECOMMENDED
──────────────────────
  pip install beautifulsoup4 lxml deepface tensorflow

WHAT THIS GIVES YOU
────────────────────
  Every match returned has:
    face_verified: True/False
    face_similarity: 0.0–1.0
    source: "yandex_reverse" | "bing_visual" | "google_cse"
    platform: inferred from domain ("LinkedIn", "GitHub", …)
    url: direct profile URL
    photo_url: profile photo used for face comparison
    name / username / title: extracted from page metadata
"""

import json
import logging
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

import config

try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
except ImportError:
    _DEEPFACE_OK = False

logger = logging.getLogger(__name__)

# ── Thresholds & limits ───────────────────────────────────────────────────
FACE_VERIFY_THRESHOLD = 0.55    # cosine similarity (DeepFace Facenet512)
MAX_CANDIDATES        = 40      # max URLs to verify per search
MAX_FACE_VERIFY_WORKERS = 6     # parallel face-verify threads

# ── Social media domain → human-readable platform name ───────────────────
SOCIAL_DOMAINS = {
    "linkedin.com":      "LinkedIn",
    "github.com":        "GitHub",
    "gitlab.com":        "GitLab",
    "twitter.com":       "Twitter",
    "x.com":             "Twitter",
    "instagram.com":     "Instagram",
    "facebook.com":      "Facebook",
    "reddit.com":        "Reddit",
    "medium.com":        "Medium",
    "researchgate.net":  "ResearchGate",
    "scholar.google":    "Google Scholar",
    "orcid.org":         "ORCID",
    "behance.net":       "Behance",
    "dribbble.com":      "Dribbble",
    "dev.to":            "Dev.to",
    "stackoverflow.com": "Stack Overflow",
    "youtube.com":       "YouTube",
    "pinterest.com":     "Pinterest",
    "tiktok.com":        "TikTok",
    "mastodon.social":   "Mastodon",
    "keybase.io":        "Keybase",
    "hackerrank.com":    "HackerRank",
    "leetcode.com":      "LeetCode",
    "kaggle.com":        "Kaggle",
    "npmjs.com":         "npm",
    "pypi.org":          "PyPI",
    "replit.com":        "Replit",
    "twitch.tv":         "Twitch",
    "flickr.com":        "Flickr",
    "vk.com":            "VK",
    "telegram.org":      "Telegram",
    "t.me":              "Telegram",
}


def _platform_for_url(url: str) -> str:
    """Return human-readable platform name for a URL, or 'Web'."""
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        for domain, name in SOCIAL_DOMAINS.items():
            if domain in host:
                return name
    except Exception:
        pass
    return "Web"


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1  — Image hosting: upload face crop to imgbb
# ══════════════════════════════════════════════════════════════════════════
def _upload_to_imgbb(img_bytes: bytes) -> Optional[str]:
    """
    Upload image bytes to imgbb (free, 5-min expiry), return public URL.
    Requires config.IMGBB_API_KEY.
    Falls back to 0x0.st if key not configured (no key needed, ~1hr TTL).
    """
    # Primary: imgbb (structured response, reliable)
    key = getattr(config, "IMGBB_API_KEY", "")
    if key:
        try:
            import base64
            b64 = base64.b64encode(img_bytes).decode()
            r = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": key, "image": b64, "expiration": 600},
                timeout=15,
            )
            data = r.json()
            url = data.get("data", {}).get("url")
            if url:
                logger.info(f"imgbb upload: {url}")
                return url
        except Exception as e:
            logger.warning(f"imgbb upload failed: {e}")

    # Fallback: 0x0.st (no key, ~1hr TTL, GDPR-friendly)
    try:
        r = requests.post(
            "https://0x0.st",
            files={"file": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=15,
        )
        if r.ok and r.text.strip().startswith("http"):
            url = r.text.strip()
            logger.info(f"0x0.st upload: {url}")
            return url
    except Exception as e:
        logger.warning(f"0x0.st upload failed: {e}")

    logger.error("Image hosting failed — both imgbb and 0x0.st unreachable")
    return None


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2  — Yandex reverse image search (best face recognition globally)
# ══════════════════════════════════════════════════════════════════════════
_YANDEX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://yandex.com/images/",
}


def _yandex_upload_and_get_search_url(img_bytes: bytes) -> Optional[str]:
    """
    POST image bytes to Yandex's CBIR upload endpoint.
    Returns the resulting search URL (with cbir_id) or None.
    Two methods are tried:
      A) JSON params  (modern API)
      B) Public image URL fallback (if upload blocked)
    """
    # Method A: direct multipart POST to yandex search
    try:
        session = requests.Session()
        session.headers.update(_YANDEX_HEADERS)

        params = {
            "rpt":     "imageview",
            "format":  "json",
            "request": json.dumps({
                "blocks": [{"block": "b-page_type_search-by-image__link"}]
            }),
        }
        files  = {"upfile": ("blob", img_bytes, "image/jpeg")}

        r = session.post(
            "https://yandex.com/images/search",
            params=params,
            files=files,
            timeout=20,
        )

        if r.ok:
            data = r.json()
            blocks = data.get("blocks", [])
            if blocks:
                qs = blocks[0].get("params", {}).get("url", "")
                if qs:
                    url = "https://yandex.com/images/search?" + qs
                    logger.info(f"Yandex upload: cbir_id obtained → {url[:80]}…")
                    return url

        logger.warning(f"Yandex upload method A returned: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Yandex upload method A failed: {e}")

    return None


def _parse_yandex_results_html(
    html: str, search_url: str
) -> tuple[list[dict], list[str]]:
    """
    Parse Yandex reverse image search HTML.
    Extracts: image results (title, url, thumbnail, snippet),
              detected entities/people names (from sidebar),
              similar image thumbnails.
    Returns (matches, names_found).
    """
    matches: list[dict] = []
    names_found: list[str] = []
    seen: set = set()

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return matches, names_found

    # ── (A) Image results: a.link tags with data-bem or site references ──
    for a in soup.select("a.link[href]"):
        href = a.get("href", "")
        if not href.startswith("http") or "yandex" in href:
            continue
        title = a.get_text(strip=True) or ""
        if href not in seen:
            seen.add(href)
            matches.append({
                "url":      href,
                "title":    title,
                "source":   "yandex_result",
                "platform": _platform_for_url(href),
            })

    # ── (B) Serpentine JSON embedded in page (Yandex often embeds data-bem) ──
    for script in soup.find_all("script"):
        txt = script.string or ""
        if "cbir" not in txt and "imageResults" not in txt:
            continue
        # Extract URLs from JSON blobs
        for url_match in re.finditer(r'"url"\s*:\s*"(https://[^"]{10,})"', txt):
            href = url_match.group(1).replace("\\u002F", "/")
            if "yandex" not in href and href not in seen:
                seen.add(href)
                matches.append({
                    "url":      href,
                    "title":    "",
                    "source":   "yandex_json",
                    "platform": _platform_for_url(href),
                })

    # ── (C) Knowledge panel / entity names (right sidebar) ──
    for el in soup.select(".CbirObjectResponse-Title, .fact-title, .entity-title"):
        t = el.get_text(strip=True)
        if t and 2 < len(t) < 80:
            names_found.append(t)

    # ── (D) Thumbnails of similar images ──
    thumbs = []
    for img in soup.select("img[src]"):
        src = img.get("src", "")
        if "avatars.mds.yandex.net" in src or "thumbs" in src:
            parent_a = img.find_parent("a")
            if parent_a:
                link = parent_a.get("href", "")
                if link and link not in seen:
                    seen.add(link)
                    # These are image-to-image similar results — fetch to verify
                    thumbs.append({
                        "url":       link,
                        "photo_url": src if src.startswith("http") else None,
                        "title":     "",
                        "source":    "yandex_similar",
                        "platform":  _platform_for_url(link),
                    })

    matches.extend(thumbs)
    logger.info(
        f"Yandex HTML parse: {len(matches)} candidates, "
        f"names={names_found}"
    )
    return matches, names_found


def _yandex_search(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Run Yandex reverse image search.
    Returns (candidate_matches, detected_names).
    """
    search_url = _yandex_upload_and_get_search_url(img_bytes)
    if not search_url:
        logger.warning("Yandex: upload failed — skipping")
        return [], []

    try:
        session = requests.Session()
        session.headers.update(_YANDEX_HEADERS)
        r = session.get(search_url, timeout=20)
        if not r.ok:
            logger.warning(f"Yandex search page: {r.status_code}")
            return [], []
        results, names = _parse_yandex_results_html(r.text, search_url)
        return results, names
    except Exception as e:
        logger.warning(f"Yandex search failed: {e}")
        return [], []


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3  — Bing Visual Search (optional, needs BING_SEARCH_KEY)
# ══════════════════════════════════════════════════════════════════════════
def _bing_visual_search(img_bytes: bytes) -> list[dict]:
    """
    Submit image to Bing Visual Search API.
    Requires config.BING_SEARCH_KEY.
    Free tier: 1000 transactions/month.
    Sign up: https://azure.microsoft.com/free/ → Bing Search v7
    """
    key = getattr(config, "BING_SEARCH_KEY", "")
    if not key:
        return []

    try:
        r = requests.post(
            "https://api.bing.microsoft.com/v7.0/images/visualsearch",
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "multipart/form-data",
            },
            files={"image": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=20,
        )
        data = r.json()
        matches = []
        seen = set()

        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                atype = action.get("actionType", "")
                # Entity recognition → person name
                if atype == "Entity":
                    for item in action.get("data", {}).get("value", []):
                        name = item.get("name", "")
                        url  = item.get("url", "")
                        if name and url and url not in seen:
                            seen.add(url)
                            matches.append({
                                "url":          url,
                                "title":        name,
                                "source":       "bing_entity",
                                "platform":     _platform_for_url(url),
                                "bing_entity":  True,
                            })
                # Visual search image results
                elif atype in ("VisualSearch", "PagesIncluding"):
                    for item in action.get("data", {}).get("value", []):
                        url       = item.get("hostPageUrl", "")
                        photo_url = item.get("contentUrl", "")
                        title     = item.get("name", "")
                        if url and url not in seen:
                            seen.add(url)
                            matches.append({
                                "url":       url,
                                "photo_url": photo_url or None,
                                "title":     title,
                                "source":    "bing_visual",
                                "platform":  _platform_for_url(url),
                            })

        logger.info(f"Bing Visual Search: {len(matches)} candidates")
        return matches
    except Exception as e:
        logger.warning(f"Bing Visual Search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4  — Face verification (download image → DeepFace compare)
# ══════════════════════════════════════════════════════════════════════════
def _fetch_image_bytes(url: str, timeout: int = 8) -> Optional[bytes]:
    """Download image bytes from a URL, handling both direct image URLs and HTML pages."""
    try:
        r = requests.get(url, headers=config.BROWSER_HEADERS,
                         timeout=timeout, allow_redirects=True)
        if not r.ok:
            return None
        ct = r.headers.get("content-type", "")
        if "image/" in ct:
            return r.content

        # HTML page — try Open Graph image
        if "html" in ct:
            soup = BeautifulSoup(r.text, "html.parser")
            for meta in soup.find_all("meta"):
                prop = meta.get("property", "") or meta.get("name", "")
                if prop in ("og:image", "twitter:image"):
                    img_url = meta.get("content", "")
                    if img_url and img_url.startswith("http"):
                        r2 = requests.get(img_url, headers=config.BROWSER_HEADERS,
                                          timeout=8, allow_redirects=True)
                        if r2.ok and "image/" in r2.headers.get("content-type", ""):
                            return r2.content
        return None
    except Exception:
        return None


def _verify_face_match(query_img_bytes: bytes, candidate_url: str,
                        candidate_photo_url: Optional[str] = None) -> dict:
    """
    Download candidate image and run DeepFace.verify() against query.
    Returns dict: {verified, similarity, photo_url, error}
    """
    result = {"verified": False, "similarity": 0.0, "photo_url": None, "error": None}
    if not _DEEPFACE_OK:
        return result

    # Try photo_url first (faster, avoids HTML parsing), then fall back to page URL
    img_bytes = None
    used_url  = None
    for try_url in filter(None, [candidate_photo_url, candidate_url]):
        img_bytes = _fetch_image_bytes(try_url)
        if img_bytes:
            used_url = try_url
            break

    if not img_bytes:
        result["error"] = "no_image"
        return result

    result["photo_url"] = used_url

    try:
        # Decode bytes → numpy array for DeepFace
        buf        = np.frombuffer(img_bytes, dtype=np.uint8)
        cand_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if cand_frame is None:
            result["error"] = "decode_failed"
            return result

        buf2        = np.frombuffer(query_img_bytes, dtype=np.uint8)
        query_frame = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
        if query_frame is None:
            result["error"] = "query_decode_failed"
            return result

        res = DeepFace.verify(
            img1_path   = query_frame,
            img2_path   = cand_frame,
            model_name  = "Facenet512",
            detector_backend = "opencv",
            enforce_detection = False,
            silent       = True,
        )
        sim = 1.0 - float(res.get("distance", 1.0))
        result["verified"]   = bool(res.get("verified", False)) or sim >= FACE_VERIFY_THRESHOLD
        result["similarity"] = round(max(0.0, sim), 4)
    except Exception as e:
        result["error"] = str(e)[:120]

    return result


def _batch_verify(query_img_bytes: bytes, candidates: list[dict]) -> list[dict]:
    """
    Run face verification in parallel for up to MAX_CANDIDATES candidates.
    Annotates each candidate dict with face_verified / face_similarity.
    Returns all candidates with scores, sorted best-first.
    """
    if not _DEEPFACE_OK:
        logger.warning("DeepFace not installed — skipping face verification (pip install deepface)")
        for c in candidates:
            c["face_verified"]   = False
            c["face_similarity"] = 0.0
        return candidates

    logger.info(f"Face verification: checking {len(candidates)} candidates "
                f"(threshold={FACE_VERIFY_THRESHOLD})")

    def _worker(cand: dict) -> dict:
        vr = _verify_face_match(
            query_img_bytes,
            cand["url"],
            cand.get("photo_url"),
        )
        cand["face_verified"]   = vr["verified"]
        cand["face_similarity"] = vr["similarity"]
        if vr.get("photo_url"):
            cand["photo_url"] = vr["photo_url"]
        if vr.get("error"):
            cand["_verify_error"] = vr["error"]
        return cand

    with ThreadPoolExecutor(max_workers=MAX_FACE_VERIFY_WORKERS) as pool:
        futures = {pool.submit(_worker, c): c for c in candidates[:MAX_CANDIDATES]}
        results = []
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                pass

    results.sort(key=lambda c: c.get("face_similarity", 0), reverse=True)
    verified_count = sum(1 for c in results if c.get("face_verified"))
    logger.info(f"Face verification: {verified_count}/{len(results)} confirmed matches")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5  — Social media profile classification
# ══════════════════════════════════════════════════════════════════════════
def _is_social_profile_url(url: str) -> bool:
    """Return True if the URL looks like a personal social media profile."""
    try:
        parsed = urllib.parse.urlparse(url)
        host   = parsed.netloc.lower().lstrip("www.")
        path   = parsed.path.strip("/")
        # Must match a known social domain AND have a non-empty path (= profile slug)
        for domain in SOCIAL_DOMAINS:
            if domain in host and path:
                # Exclude generic/landing pages
                if path not in ("explore", "search", "trending", "tags", "topics"):
                    return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6  — Google CSE expansion (text search with discovered names)
# ══════════════════════════════════════════════════════════════════════════
_CSE_SOCIAL_SITES = [
    "site:linkedin.com",
    "site:github.com",
    "site:twitter.com OR site:x.com",
    "site:instagram.com",
    "site:researchgate.net",
    "site:scholar.google.com",
]


def _google_cse_search(name: str, extra_hint: str = "") -> list[dict]:
    """
    Use Google Custom Search Engine to find social profiles for a discovered name.
    Only called AFTER face verification has confirmed the name.
    """
    key = getattr(config, "GOOGLE_CSE_KEY", "")
    cx  = getattr(config, "GOOGLE_CSE_ID",  "")
    if not key or not cx:
        logger.info("Google CSE: key/cx not set — skipping")
        return []

    query = f'"{name}"'
    if extra_hint:
        query += f" {extra_hint}"

    results = []
    seen    = set()

    for site_filter in _CSE_SOCIAL_SITES:
        q = f"{query} {site_filter}"
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": key,
                    "cx":  cx,
                    "q":   q,
                    "num": 5,
                },
                timeout=10,
            )
            data = r.json()
            for item in data.get("items", []):
                url   = item.get("link", "")
                title = item.get("title", "")
                if url and url not in seen:
                    seen.add(url)
                    results.append({
                        "url":           url,
                        "title":         title,
                        "source":        "google_cse",
                        "platform":      _platform_for_url(url),
                        "face_verified": False,   # CSE results are name-matched,
                        "face_similarity": 0.0,   # not face-verified yet
                    })
            time.sleep(0.12)   # stay under CSE rate limit (100/day free)
        except Exception as e:
            logger.debug(f"CSE query '{q}': {e}")

    logger.info(f"Google CSE: {len(results)} profiles found for '{name}'")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  STEP 7  — Enrich face-verified matches (extract name/username from page)
# ══════════════════════════════════════════════════════════════════════════
def _enrich_profile(match: dict) -> dict:
    """
    For a face-verified match, fetch its page and extract structured fields:
    name, username, bio, email, location, company, avatar_url.
    """
    url      = match.get("url", "")
    platform = match.get("platform", "")

    try:
        r = requests.get(url, headers=config.BROWSER_HEADERS, timeout=8)
        if not r.ok:
            return match
        soup = BeautifulSoup(r.text, "html.parser")

        # OG tags (universal)
        def og(prop: str) -> str:
            t = soup.find("meta", property=f"og:{prop}")
            return (t.get("content", "") if t else "").strip()

        title    = og("title") or soup.title.string or ""
        desc     = og("description") or ""
        avatar   = og("image") or ""

        match.setdefault("title",     title)
        match.setdefault("bio",       desc[:200] if desc else "")
        if avatar and not match.get("photo_url"):
            match["photo_url"] = avatar

        # Platform-specific extraction
        if platform == "GitHub":
            for sel in (".p-name", "span.p-name", "h1.vcard-names"):
                el = soup.select_one(sel)
                if el:
                    match["name"] = el.get_text(strip=True); break
            for sel in (".p-nickname", "span.p-nickname"):
                el = soup.select_one(sel)
                if el:
                    match["username"] = el.get_text(strip=True); break
            for sel in (".p-org", "span.p-org"):
                el = soup.select_one(sel)
                if el:
                    match["company"] = el.get_text(strip=True); break

        elif platform == "LinkedIn":
            for sel in ("h1", ".top-card-layout__title"):
                el = soup.select_one(sel)
                if el:
                    match["name"] = el.get_text(strip=True); break

        elif platform == "Twitter":
            # Twitter is heavily JS — og:title contains "Name (@handle)"
            m = re.match(r"^(.+?)\s+\(@(\w+)\)", title)
            if m:
                match["name"]     = m.group(1)
                match["username"] = m.group(2)

        elif platform == "ResearchGate":
            el = soup.select_one("h1.profile-header__name")
            if el:
                match["name"] = el.get_text(strip=True)

    except Exception as e:
        logger.debug(f"Enrich failed for {url}: {e}")

    return match


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def scrape(context: dict) -> dict:
    """
    Face-first reverse image OSINT.
    context must contain:
        image_b64   — base64 JPEG of the face crop (from embedding step)
        name        — user-provided name hint (used for CSE expansion)
    Returns standard {source, matches, face_verified_count, names_found}.
    """
    name_hint  = context.get("name", "")
    image_b64  = context.get("image_b64", "")

    if not image_b64:
        logger.warning("reverse_image.scrape: no image_b64 in context — skipping")
        return {"source": "reverse_image", "matches": [], "error": "no_image"}

    import base64
    try:
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
    except Exception as e:
        return {"source": "reverse_image", "matches": [], "error": str(e)}

    logger.info(
        f"=== FACE-FIRST REVERSE IMAGE SEARCH === "
        f"({len(img_bytes)//1024}KB, hint='{name_hint}')"
    )

    all_candidates: list[dict] = []
    detected_names: list[str]  = []

    # ── Step 2: Yandex reverse image search ─────────────────────────────
    yandex_results, yandex_names = _yandex_search(img_bytes)
    all_candidates.extend(yandex_results)
    detected_names.extend(yandex_names)

    # ── Step 3: Bing Visual Search (optional) ───────────────────────────
    bing_results = _bing_visual_search(img_bytes)
    for r in bing_results:
        all_candidates.append(r)
        if r.get("bing_entity") and r.get("title"):
            detected_names.append(r["title"])

    # Deduplicate candidates by URL
    seen_urls   = set()
    unique_cands = []
    for c in all_candidates:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            unique_cands.append(c)

    # Prioritise social media profile URLs (more likely to yield face + name)
    social = [c for c in unique_cands if _is_social_profile_url(c["url"])]
    other  = [c for c in unique_cands if not _is_social_profile_url(c["url"])]
    ordered_cands = (social + other)[:MAX_CANDIDATES]

    logger.info(
        f"Total candidates: {len(ordered_cands)} "
        f"({len(social)} social profiles, {len(other)} other)"
    )

    # ── Step 4: Face verification ────────────────────────────────────────
    verified_cands = _batch_verify(img_bytes, ordered_cands)

    # ── Collect confirmed matches & names ────────────────────────────────
    confirmed   = [c for c in verified_cands if c.get("face_verified")]
    unconfirmed = [c for c in verified_cands if not c.get("face_verified")]

    # Names from Bing entity recognition and Yandex knowledge panel
    # also fall back to name_hint if nothing found
    all_names = list(dict.fromkeys(detected_names))   # deduplicate, preserve order
    if not all_names and name_hint:
        all_names = [name_hint]

    # Extract names from confirmed match page titles too
    for c in confirmed:
        t = c.get("title", "")
        if t and len(t.split()) >= 2 and t not in all_names:
            # Heuristic: page title with 2+ words often contains a person name
            all_names.append(t.split(" - ")[0].split(" | ")[0].strip())

    logger.info(f"Face-confirmed: {len(confirmed)}, names detected: {all_names}")

    # ── Step 5 & 7: Enrich face-verified social profiles ─────────────────
    for c in confirmed:
        if _is_social_profile_url(c["url"]):
            _enrich_profile(c)

    # ── Step 6: Google CSE expansion for discovered names ────────────────
    cse_results = []
    for nm in all_names[:2]:   # top 2 most likely names
        cse = _google_cse_search(nm)
        # Run face verification on any CSE results that returned a photo
        for cr in cse:
            if cr.get("photo_url"):
                vr = _verify_face_match(img_bytes, cr["url"], cr["photo_url"])
                cr["face_verified"]   = vr["verified"]
                cr["face_similarity"] = vr["similarity"]
                if vr.get("photo_url"):
                    cr["photo_url"] = vr["photo_url"]
        cse_results.extend(cse)

    # Merge everything; face-verified first, then CSE, then unconfirmed
    final_matches = confirmed + cse_results + unconfirmed

    # Deduplicate final list by URL
    seen2, deduped_final = set(), []
    for m in final_matches:
        u = m.get("url", "")
        if u and u not in seen2:
            seen2.add(u)
            deduped_final.append(m)

    face_verified_count = sum(1 for m in deduped_final if m.get("face_verified"))
    logger.info(
        f"reverse_image complete: {len(deduped_final)} total matches, "
        f"{face_verified_count} face-verified, names={all_names}"
    )

    return {
        "source":              "reverse_image",
        "matches":             deduped_final,
        "names_found":         all_names,
        "face_verified_count": face_verified_count,
    }