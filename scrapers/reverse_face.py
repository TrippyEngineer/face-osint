import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

FACE_POSSIBLE  = getattr(config, "FACE_POSSIBLE",  0.50)
MAX_CANDIDATES = 25   # 25 × ~1s/face ≈ 20s verify; keeps total under 90s timeout
VERIFY_WORKERS = 6

# Engine-cached thumbnail hosts. These serve a copy of the QUERY image the engine
# matched, so face-verifying the query against them is self-referential (~0.94 for
# anything) AND showing them returns the user's own image. We verify/display the
# candidate page's OWN photo instead.
_ENGINE_THUMB_HOSTS = (
    "encrypted-tbn0.gstatic.com", "encrypted-tbn1.gstatic.com",
    "encrypted-tbn2.gstatic.com", "encrypted-tbn3.gstatic.com",
    "gstatic.com/images", "googleusercontent.com/gps-proxy",
    "mm.bing.net/th", "bing.com/th", "tse1.mm.bing.net", "tse2.mm.bing.net",
    "tse3.mm.bing.net", "tse4.mm.bing.net", "yandex.net/i?", "avatars.mds.yandex.net",
)

def _is_engine_thumbnail(u: str) -> bool:
    u = (u or "").lower()
    return any(h in u for h in _ENGINE_THUMB_HOSTS)

def _name_sim(a: str, b: str) -> float:
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

_SOCIAL = {
    "linkedin.com": "LinkedIn", "github.com": "GitHub", "gitlab.com": "GitLab",
    "twitter.com": "Twitter", "x.com": "Twitter", "instagram.com": "Instagram",
    "facebook.com": "Facebook", "reddit.com": "Reddit", "medium.com": "Medium",
    "researchgate.net": "ResearchGate", "orcid.org": "ORCID",
    "behance.net": "Behance", "dribbble.com": "Dribbble", "dev.to": "Dev.to",
    "stackoverflow.com": "Stack Overflow", "youtube.com": "YouTube",
    "tiktok.com": "TikTok", "pinterest.com": "Pinterest", "vk.com": "VK",
    "hackerrank.com": "HackerRank", "kaggle.com": "Kaggle",
    "replit.com": "Replit", "twitch.tv": "Twitch", "flickr.com": "Flickr",
    "t.me": "Telegram", "keybase.io": "Keybase",
}

def _platform(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        for d, n in _SOCIAL.items():
            if d in host:
                return n
    except Exception:
        pass
    return "Web"


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE HOSTING
# ══════════════════════════════════════════════════════════════════════════
def _host_image(img_bytes: bytes) -> Optional[str]:
    """Upload face crop to a public URL. SerpApi needs this."""

    # 1. imgbb — API key set in .env
    key = getattr(config, "IMGBB_API_KEY", "")
    if key:
        try:
            r = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": key, "image": base64.b64encode(img_bytes).decode(),
                      "expiration": 600},
                timeout=12,
            )
            url = r.json().get("data", {}).get("url")
            if url:
                logger.info(f"imgbb OK → {url[:60]}")
                return url
        except Exception as e:
            logger.debug(f"imgbb: {e}")

    # 2. catbox.moe — no key needed, very reliable
    try:
        r = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload", "userhash": ""},
            files={"fileToUpload": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=15,
        )
        if r.ok and r.text.strip().startswith("https://"):
            url = r.text.strip()
            logger.info(f"catbox.moe OK → {url}")
            return url
    except Exception as e:
        logger.debug(f"catbox: {e}")

    # 3. litterbox — no key, 1hr TTL
    try:
        r = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "1h"},
            files={"fileToUpload": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=15,
        )
        if r.ok and r.text.strip().startswith("https://"):
            url = r.text.strip()
            logger.info(f"litterbox OK → {url}")
            return url
    except Exception as e:
        logger.debug(f"litterbox: {e}")

    logger.warning("All image hosting failed — SerpApi engines will be skipped")
    return None


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 1 — SerpApi Google Lens
# ══════════════════════════════════════════════════════════════════════════
def _serpapi_lens(public_url: Optional[str]) -> tuple[list[dict], list[str]]:
    """
    Google Lens via SerpApi.
    Docs: https://serpapi.com/google-lens-api
    Returns visual matches + entity panel (person name if known to Google).
    """
    key = getattr(config, "SERPAPI_KEY", "")
    if not key:
        logger.info("SerpApi: no SERPAPI_KEY")
        return [], []
    if not public_url:
        logger.info("SerpApi Lens: no public URL (image hosting failed)")
        return [], []

    matches, names, seen = [], [], set()

    for search_type in ["visual_matches", "exact_matches"]:
        try:
            r = requests.get(
                "https://serpapi.com/search",
                params={
                    "api_key": key,
                    "engine":  "google_lens",
                    "url":     public_url,
                    "type":    search_type,
                    "hl":      "en",
                },
                timeout=25,
            )
            if not r.ok:
                logger.warning(f"SerpApi Lens {search_type}: HTTP {r.status_code}")
                continue
            data = r.json()

            for item in data.get("visual_matches", []):
                url = item.get("link", "")
                if url and url not in seen:
                    seen.add(url)
                    matches.append({
                        "url":       url,
                        "title":     item.get("title", ""),
                        "photo_url": item.get("thumbnail") or None,
                        "platform":  _platform(url),
                        "source":    f"serpapi_lens_{search_type}",
                    })

            # Knowledge graph / entity panel — Lens sometimes identifies the person
            # directly. SerpApi returns this as a DICT for Lens (a list for some
            # engines); iterating a dict as a list yielded keys/raised and was
            # swallowed, which is why entity names never extracted. Handle both.
            kg_data  = data.get("knowledge_graph")
            kg_items = (kg_data if isinstance(kg_data, list)
                        else [kg_data] if isinstance(kg_data, dict) else [])
            for kg in kg_items:
                if not isinstance(kg, dict):
                    continue
                nm  = kg.get("title", "") or kg.get("name", "")
                url = kg.get("link", "")
                if nm and len(nm.split()) >= 2:
                    names.append(nm)
                    logger.info(f"SerpApi Lens entity: '{nm}'")
                if url and url not in seen:
                    seen.add(url)
                    matches.append({
                        "url":      url,
                        "title":    nm,
                        "platform": _platform(url),
                        "source":   "serpapi_lens_entity",
                    })

            time.sleep(0.2)  # avoid hammering quota
        except Exception as e:
            logger.debug(f"SerpApi Lens {search_type}: {e}")

    logger.info(f"SerpApi Google Lens: {len(matches)} candidates, names={names}")
    return matches, names


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 2 — SerpApi Yandex Reverse Image
# ══════════════════════════════════════════════════════════════════════════
def _serpapi_yandex(public_url: Optional[str]) -> tuple[list[dict], list[str]]:
    """
    Yandex Images via SerpApi.
    Docs: https://serpapi.com/yandex-search-api (images)
    Best for Indian/Eastern European/South Asian content.
    """
    key = getattr(config, "SERPAPI_KEY", "")
    if not key or not public_url:
        return [], []

    matches, names, seen = [], [], set()
    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={
                "api_key": key,
                "engine":  "yandex_images",
                "url":     public_url,
            },
            timeout=25,
        )
        if not r.ok:
            logger.warning(f"SerpApi Yandex: HTTP {r.status_code}")
            return [], []
        data = r.json()

        for item in data.get("image_results", []):
            url = item.get("link", "")
            if url and url not in seen:
                seen.add(url)
                thumb = item.get("thumbnail", "")
                if isinstance(thumb, dict):
                    thumb = thumb.get("src", "")
                matches.append({
                    "url":       url,
                    "title":     item.get("title", ""),
                    "photo_url": thumb or None,
                    "platform":  _platform(url),
                    "source":    "serpapi_yandex",
                })

        kg = data.get("knowledge_graph", {})
        if isinstance(kg, dict):
            nm = kg.get("title", "") or kg.get("name", "")
            if nm and len(nm.split()) >= 2:
                names.append(nm)
                logger.info(f"SerpApi Yandex entity: '{nm}'")

        logger.info(f"SerpApi Yandex: {len(matches)} candidates, names={names}")
        return matches, names
    except Exception as e:
        logger.warning(f"SerpApi Yandex: {e}")
        return [], []


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 3 — Yandex CBir direct upload (no key, corrected 2025 endpoint)
# ══════════════════════════════════════════════════════════════════════════
_YA_HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://yandex.com/images/",
}

def _yandex_direct(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    """Direct POST to Yandex CBIR — no key needed, no hosting needed."""
    for block in [
        '{"blocks":[{"block":"cbir-uploader__get-cbir-id"}]}',
        '{"blocks":[{"block":"b-page_type_search-by-image__link"}]}',
    ]:
        try:
            r = requests.post(
                "https://yandex.com/images/search",
                params={"rpt": "imageview", "format": "json", "request": block},
                files={"upfile": ("blob", img_bytes, "image/jpeg")},
                headers=_YA_HDR,
                timeout=20,
            )
            if not r.ok:
                continue
            blks = r.json().get("blocks", [])
            if not blks:
                continue
            qs = blks[0].get("params", {}).get("url", "")
            if not qs:
                continue

            s  = requests.Session()
            s.headers.update(_YA_HDR)
            r2 = s.get(f"https://yandex.com/images/search?{qs}", timeout=20)
            if not r2.ok:
                continue

            matches, names = _parse_yandex_html(r2.text)
            logger.info(f"Yandex direct: {len(matches)} candidates, names={names}")
            return matches, names
        except Exception as e:
            logger.debug(f"Yandex direct block attempt: {e}")

    logger.warning("Yandex direct upload: both block names failed")
    return [], []

def _parse_yandex_html(html: str) -> tuple[list[dict], list[str]]:
    matches, names, seen = [], [], set()
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for item in soup.select(".CbirSites-Item, .cbir-section__sites-item"):
        a = item.select_one("a[href]")
        if not a:
            continue
        href = a.get("href", "")
        if href.startswith("http") and "yandex" not in href and href not in seen:
            seen.add(href)
            img_t = item.select_one("img")
            thumb = img_t.get("src", "") if img_t else ""
            matches.append({
                "url":       href,
                "title":     a.get_text(strip=True)[:120],
                "photo_url": thumb if thumb.startswith("http") else None,
                "platform":  _platform(href),
                "source":    "yandex_direct",
            })

    for script in soup.find_all("script"):
        txt = script.string or ""
        if not txt or "cbir" not in txt:
            continue
        for m in re.finditer(r'"(?:url|pageUrl)"\s*:\s*"(https?://[^"]{10,})"', txt):
            href = m.group(1).replace("\\u002F", "/")
            if "yandex" not in href and href not in seen:
                seen.add(href)
                matches.append({"url": href, "title": "",
                                 "platform": _platform(href), "source": "yandex_json"})

    for el in soup.select(".CbirObjectResponse-Title, .CbirPeople-Title, .entity-title"):
        t = el.get_text(strip=True)
        if t and 3 < len(t) < 80:
            names.append(t)

    return matches, names


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 4 — Google CSE image + social search
# ══════════════════════════════════════════════════════════════════════════
def _google_cse(name_hint: str, public_url: Optional[str]) -> list[dict]:
    key = getattr(config, "GOOGLE_CSE_KEY", "")
    cx  = getattr(config, "GOOGLE_CSE_ID",  "")
    if not key or not cx:
        return []

    matches, seen = [], set()

    # Image search by URL
    if public_url:
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": key, "cx": cx, "searchType": "image",
                        "imgType": "face", "imgUrl": public_url, "num": 10},
                timeout=12,
            )
            for item in r.json().get("items", []):
                url = item.get("link", "")
                if url and url not in seen:
                    seen.add(url)
                    matches.append({
                        "url":       url,
                        "title":     item.get("title", ""),
                        "photo_url": item.get("image", {}).get("thumbnailLink"),
                        "platform":  _platform(url),
                        "source":    "cse_image",
                    })
        except Exception as e:
            logger.debug(f"CSE image: {e}")

    # Social profile text search
    if name_hint:
        for site in [
            "site:linkedin.com/in",
            "site:github.com",
            "site:instagram.com",
            "site:twitter.com OR site:x.com",
            "site:researchgate.net",
        ]:
            try:
                r = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"key": key, "cx": cx, "num": 5,
                            "q": f'"{name_hint}" {site}'},
                    timeout=10,
                )
                for item in r.json().get("items", []):
                    url = item.get("link", "")
                    if url and url not in seen:
                        seen.add(url)
                        matches.append({
                            "url":      url,
                            "title":    item.get("title", ""),
                            "snippet":  item.get("snippet", "")[:200],
                            "platform": _platform(url),
                            "source":   "cse_social",
                        })
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"CSE {site}: {e}")

    logger.info(f"Google CSE: {len(matches)} candidates")
    return matches


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 5 — PimEyes (paid API stub)
# ══════════════════════════════════════════════════════════════════════════
def _search_pimeyes(image_b64: str) -> list:
    """Search PimEyes API. Returns [] if key not set or API fails."""
    try:
        key = getattr(config, "PIMEYES_API_KEY", "")
        if not key:
            return []
        r = requests.post(
            "https://pimeyes.com/api/search/advanced",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": config.BROWSER_HEADERS["User-Agent"],
            },
            json={"image": image_b64},
            timeout=20,
        )
        if not r.ok:
            logger.debug(f"PimEyes: HTTP {r.status_code}")
            return []
        data = r.json()
        matches = []
        for hit in data.get("results", [])[:20]:
            url        = hit.get("url", "") or hit.get("pageUrl", "")
            thumb      = hit.get("thumbnailUrl", "") or hit.get("thumbnail", "")
            title      = hit.get("name", "") or hit.get("title", "")
            if not url:
                continue
            matches.append({
                "source":      "pimeyes",
                "url":         url,
                "preview_url": thumb or None,
                "snippet":     title,
                "platform":    _platform(url),
            })
        logger.info(f"PimEyes: {len(matches)} results")
        return matches
    except Exception as e:
        logger.debug(f"PimEyes: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
#  FACE VERIFICATION (DeepFace inline)
# ══════════════════════════════════════════════════════════════════════════
def _http_image(try_url: str) -> tuple[Optional[bytes], Optional[str]]:
    """Fetch raw image bytes from a direct image URL or a page's og:image.
    Returns (bytes, actual_image_url) or (None, None)."""
    try:
        r = requests.get(try_url, headers=config.BROWSER_HEADERS,
                         timeout=8, allow_redirects=True)
        if not r.ok:
            return None, None
        ct = r.headers.get("content-type", "")
        if "image/" in ct:
            return r.content, try_url
        if "html" in ct:
            soup = BeautifulSoup(r.text, "html.parser")
            for meta in soup.find_all("meta"):
                prop = meta.get("property", "") or meta.get("name", "")
                if prop in ("og:image", "twitter:image"):
                    img_url = meta.get("content", "")
                    if img_url and img_url.startswith("http"):
                        r2 = requests.get(img_url, headers=config.BROWSER_HEADERS, timeout=6)
                        if r2.ok and "image/" in r2.headers.get("content-type", ""):
                            return r2.content, img_url
    except Exception:
        pass
    return None, None


def _fetch_image(url: str, photo_url: Optional[str] = None
                 ) -> tuple[Optional[bytes], Optional[str], bool]:
    """Fetch the candidate's REAL photo for face-verify, preferring the page's own
    image over an engine-cached thumbnail (a copy of the QUERY → self-referential
    ~0.94 match). Returns (bytes, image_url, used_thumbnail)."""
    thumb      = photo_url if (photo_url and _is_engine_thumbnail(photo_url)) else None
    real_photo = photo_url if (photo_url and not _is_engine_thumbnail(photo_url)) else None
    # Prefer a real direct photo_url → the page (og:image) → (last resort) the thumb.
    for try_url in filter(None, [real_photo, url]):
        data, img_url = _http_image(try_url)
        if data:
            return data, img_url, False
    if thumb:
        data, img_url = _http_image(thumb)
        if data:
            return data, img_url, True
    return None, None, False


def _verify_one(query_np: np.ndarray, cand: dict) -> dict:
    cand.setdefault("face_verified", False)
    cand.setdefault("face_score", None)
    cand.setdefault("face_similarity", 0.0)

    img_bytes, img_url, used_thumb = _fetch_image(cand.get("url", ""), cand.get("photo_url"))
    if used_thumb or not img_bytes:
        # Only the engine's cached thumbnail of the QUERY was reachable (or no image
        # at all). Verifying the query against its own cached copy is self-referential
        # (~0.94 for anything), so record NO face score and drop the thumbnail so the
        # UI never shows the user's own query image back as a "match".
        if used_thumb:
            cand["face_evidence"] = "thumbnail_only"
            cand["photo_url"]     = None
        else:
            cand["_no_image"] = True
        return cand
    try:
        buf        = np.frombuffer(img_bytes, dtype=np.uint8)
        cand_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if cand_frame is None:
            return cand
        from deepface import DeepFace
        res = DeepFace.verify(
            img1_path=query_np, img2_path=cand_frame,
            model_name="Facenet512", detector_backend="opencv",
            enforce_detection=False, silent=True,
        )
        sim = round(max(0.0, 1.0 - float(res.get("distance", 1.0))), 4)
        cand["face_similarity"] = sim
        cand["face_score"]      = sim
        cand["face_verified"]   = sim >= FACE_POSSIBLE
        cand["face_evidence"]   = "page_photo"
        cand["no_face_photo"]   = False
        if img_url:
            cand["photo_url"] = img_url   # show the page's real photo, not the cached thumb
    except Exception as e:
        logger.debug(f"verify {cand.get('url','')[:60]}: {e}")
    return cand


def _batch_verify(query_np: np.ndarray, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    logger.info(f"Face-verify: {len(candidates)} candidates (threshold={FACE_POSSIBLE})")
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
        results = [f.result() for f in as_completed(
            [pool.submit(_verify_one, query_np, c) for c in candidates[:MAX_CANDIDATES]]
        )]
    results.sort(key=lambda c: c.get("face_similarity", 0), reverse=True)
    confirmed = sum(1 for c in results if c.get("face_verified"))
    logger.info(f"Face-verify: {confirmed}/{len(results)} confirmed")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  IDENTITY ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════
def _enrich(match: dict) -> dict:
    url      = match.get("url", "")
    platform = match.get("platform", "")
    try:
        r = requests.get(url, headers=config.BROWSER_HEADERS, timeout=8)
        if not r.ok:
            return match
        soup = BeautifulSoup(r.text, "html.parser")

        def og(p):
            t = soup.find("meta", property=f"og:{p}")
            return (t.get("content", "") if t else "").strip()

        title  = og("title") or (soup.title.string if soup.title else "") or ""
        avatar = og("image") or ""
        desc   = og("description") or ""
        match.setdefault("title", title[:140])
        match.setdefault("bio",   desc[:200])
        if avatar and not match.get("photo_url"):
            match["photo_url"] = avatar

        if platform == "GitHub":
            for sel in (".p-name",):
                el = soup.select_one(sel)
                if el: match["name"] = el.get_text(strip=True); break
            for sel in (".p-nickname",):
                el = soup.select_one(sel)
                if el: match["username"] = el.get_text(strip=True); break
            uname = match.get("username", "")
            if uname and getattr(config, "GITHUB_TOKEN", ""):
                try:
                    gh = requests.get(
                        f"https://api.github.com/users/{uname}",
                        headers={"Authorization": f"token {config.GITHUB_TOKEN}"},
                        timeout=8,
                    ).json()
                    for k in ("name", "bio", "company", "location", "email"):
                        if gh.get(k): match.setdefault(k, gh[k])
                    if gh.get("avatar_url"):
                        match["photo_url"] = gh["avatar_url"]
                except Exception:
                    pass

        if not match.get("name") and title:
            cand = re.split(r"\s*[\|—–\-]\s*", title)[0].strip()
            if 2 <= len(cand.split()) <= 5:
                match.setdefault("name", cand)
    except Exception as e:
        logger.debug(f"enrich {url}: {e}")
    return match


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 6 — Playwright Google Lens (free, no API key, no image hosting)
# ══════════════════════════════════════════════════════════════════════════
def _playwright_google_lens(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    """Google Lens via headless Chromium. No SerpApi key needed.

    Runs the Playwright ASYNC API in this worker thread's own event loop via
    asyncio.run() — the supported way to drive Playwright from a thread. The old
    sync API used a process-global driver that intermittently raised
    'This event loop is already running' when Lens + Bing ran in parallel."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        logger.info("playwright not installed — skipping Lens (pip install playwright && playwright install chromium)")
        return [], []
    try:
        return asyncio.run(_lens_async(img_bytes))
    except Exception as e:
        logger.warning(f"Playwright Google Lens: {e}")
        return [], []


async def _lens_async(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    from playwright.async_api import async_playwright

    matches, names, seen = [], [], set()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            ctx = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = await ctx.new_page()
            page.set_default_timeout(15000)

            # Google Images homepage → click camera icon (visual search)
            await page.goto("https://images.google.com/", timeout=15000, wait_until="domcontentloaded")
            await page.click('[aria-label="Search by image"]', timeout=8000)
            await page.wait_for_timeout(800)

            # Intercept the native OS file dialog that "upload a file" triggers
            async with page.expect_file_chooser(timeout=8000) as fc_info:
                await page.get_by_role("button", name="upload a file").click(timeout=6000)
            chooser = await fc_info.value
            await chooser.set_files(tmp_path)

            # Wait for results page
            await page.wait_for_url("**search**", timeout=25000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            # Collect all external result links
            for a in (await page.locator("a[href]").all())[:120]:
                try:
                    href = await a.get_attribute("href") or ""
                    if (
                        href.startswith("http")
                        and "google.com" not in href
                        and "gstatic.com" not in href
                        and href not in seen
                    ):
                        seen.add(href)
                        title, thumb = "", None
                        try:
                            title = (await a.inner_text(timeout=400)).strip()[:120]
                        except Exception:
                            pass
                        try:
                            src = await a.locator("img").first.get_attribute("src", timeout=400)
                            if src and src.startswith("http"):
                                thumb = src
                        except Exception:
                            pass
                        matches.append({
                            "url":       href,
                            "title":     title,
                            "photo_url": thumb,
                            "platform":  _platform(href),
                            "source":    "playwright_google_lens",
                        })
                except Exception:
                    continue

            # Knowledge panel entity name (skip generic UI labels)
            _SKIP = {"ai overview", "more images", "search results", "all results", "images"}
            for sel in [
                '[data-attrid="title"] span',
                ".kno-ecr-pt span",
                ".dAassd span",
                '[aria-level="2"]',
            ]:
                try:
                    el = page.locator(sel).first
                    if await el.is_visible(timeout=800):
                        nm = (await el.inner_text(timeout=400)).strip()
                        if (nm and 2 <= len(nm.split()) <= 5
                                and nm not in names
                                and nm.lower() not in _SKIP):
                            names.append(nm)
                            logger.info(f"Playwright Lens entity: '{nm}'")
                            break
                except Exception:
                    continue

            await browser.close()

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    logger.info(f"Playwright Google Lens: {len(matches)} candidates, names={names}")
    return matches, names


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE 7 — Playwright Bing Visual Search (free, complements Google Lens)
# ══════════════════════════════════════════════════════════════════════════
def _playwright_bing_visual(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    """Bing Visual Search via headless Chromium. No API key needed.
    Different index from Google — useful for Eastern Europe / Central Asia / India.
    Async API run via asyncio.run() in this thread's own loop (see Lens)."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return [], []
    try:
        return asyncio.run(_bing_async(img_bytes))
    except Exception as e:
        logger.warning(f"Playwright Bing Visual: {e}")
        return [], []


async def _bing_async(img_bytes: bytes) -> tuple[list[dict], list[str]]:
    from playwright.async_api import async_playwright

    matches, names, seen = [], [], set()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = await ctx.new_page()
            page.set_default_timeout(15000)

            # Bing Images → camera icon → upload panel → file chooser
            await page.goto("https://www.bing.com/images", timeout=15000, wait_until="domcontentloaded")
            await page.get_by_role("button", name="Search using an image").click(timeout=6000)
            await page.wait_for_timeout(600)

            # Intercept native file dialog from "upload an image" button
            async with page.expect_file_chooser(timeout=8000) as fc_info:
                await page.get_by_role("button", name="upload an image").click(timeout=6000)
            chooser = await fc_info.value
            await chooser.set_files(tmp_path)

            await page.wait_for_load_state("networkidle", timeout=20000)

            # Bing result cards — each <a m='{"purl":"...","murl":"..."}'>
            for a in (await page.locator("a[m]").all())[:80]:
                try:
                    m_attr = await a.get_attribute("m") or "{}"
                    m_data = json.loads(m_attr)
                    actual_url = m_data.get("purl") or m_data.get("murl") or ""
                    if actual_url and "bing.com" not in actual_url and actual_url not in seen:
                        seen.add(actual_url)
                        title = (await a.get_attribute("title") or "").strip()[:120]
                        thumb = m_data.get("turl") or None
                        matches.append({
                            "url":       actual_url,
                            "title":     title,
                            "photo_url": thumb,
                            "platform":  _platform(actual_url),
                            "source":    "playwright_bing_visual",
                        })
                except Exception:
                    continue

            # Fallback: collect plain href links if m-attribute cards are absent
            if not matches:
                for a in (await page.locator("a[href]").all())[:80]:
                    try:
                        href = await a.get_attribute("href") or ""
                        if href.startswith("http") and "bing.com" not in href and href not in seen:
                            seen.add(href)
                            title = (await a.get_attribute("title") or await a.inner_text(timeout=300) or "").strip()[:120]
                            matches.append({
                                "url": href, "title": title,
                                "platform": _platform(href), "source": "playwright_bing_visual",
                            })
                    except Exception:
                        continue

            # Bing entity panel
            for sel in [".b_entityTitle", ".va_title h1", "[class*='entity'] h1"]:
                try:
                    el = page.locator(sel).first
                    if await el.is_visible(timeout=800):
                        nm = (await el.inner_text(timeout=400)).strip()
                        if nm and 2 <= len(nm.split()) <= 5 and nm not in names:
                            names.append(nm)
                            logger.info(f"Playwright Bing entity: '{nm}'")
                            break
                except Exception:
                    continue

            await browser.close()

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    logger.info(f"Playwright Bing Visual: {len(matches)} candidates, names={names}")
    return matches, names


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def scrape(context: dict) -> dict:
    name_hint = context.get("name", "")
    image_b64 = context.get("image_b64", "")

    if not image_b64:
        return {"source": "reverse_face", "matches": [], "error": "no image_b64 in context"}

    try:
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
    except Exception as e:
        return {"source": "reverse_face", "matches": [], "error": str(e)}

    buf      = np.frombuffer(img_bytes, dtype=np.uint8)
    query_np = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if query_np is None:
        return {"source": "reverse_face", "matches": [], "error": "cv2 decode failed"}

    logger.info(
        f"=== FACE-FIRST ENGINE === hint='{name_hint}' "
        f"{query_np.shape[1]}x{query_np.shape[0]}px"
    )

    # Upload face to public URL first (needed by SerpApi)
    public_url = _host_image(img_bytes)

    all_candidates: list[dict] = []
    detected_names: list[str]  = []

    # All engines in parallel — Playwright engines (6 + 7) run alongside existing ones.
    # NOTE: _yandex_direct (keyless CBIR upload) was dropped — Yandex anti-bot blocks
    # it ("both block names failed"), so it only added latency with zero results.
    # SerpApi Yandex (_serpapi_yandex) is kept; it's a clean no-op without SERPAPI_KEY.
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_sl  = pool.submit(_serpapi_lens,          public_url)
        f_sy  = pool.submit(_serpapi_yandex,        public_url)
        f_cse = pool.submit(_google_cse,            name_hint, public_url)
        f_pl  = pool.submit(_playwright_google_lens, img_bytes)
        f_pb  = pool.submit(_playwright_bing_visual, img_bytes)

        sl_res, sl_names = f_sl.result()
        sy_res, sy_names = f_sy.result()
        cse_res           = f_cse.result()
        pl_res, pl_names  = f_pl.result()
        pb_res, pb_names  = f_pb.result()

    all_candidates += sl_res + sy_res + cse_res + pl_res + pb_res
    detected_names  += sl_names + sy_names + pl_names + pb_names

    # Engine 5 — PimEyes (paid, silently skipped if key not set)
    try:
        pimeyes_res = _search_pimeyes(image_b64)
        all_candidates += pimeyes_res
    except Exception as e:
        logger.debug(f"PimEyes wrapper: {e}")

    # Deduplicate
    seen, unique = set(), []
    for c in all_candidates:
        u = c.get("url", "")
        if u and u not in seen:
            seen.add(u); unique.append(c)

    # Social profiles first (more likely to have face photo + identity)
    social  = [c for c in unique if _platform(c["url"]) != "Web"]
    other   = [c for c in unique if _platform(c["url"]) == "Web"]
    ordered = (social + other)[:MAX_CANDIDATES]

    logger.info(f"Candidates: {len(ordered)} ({len(social)} social, {len(other)} other)")

    # Face verification
    verified    = _batch_verify(query_np, ordered)
    confirmed   = [c for c in verified if c.get("face_verified")]
    unconfirmed = [c for c in verified if not c.get("face_verified")]

    # Collect names from confirmed results — but only adopt a page-title name when
    # it CORROBORATES the query name. Without this gate a look-alike's page title
    # (e.g. "PUJYA GHOSH" for query "Neeraj Jain") became the identity and drove a
    # wrong CSE name-expansion. With no query name, don't invent one from a title.
    for c in confirmed:
        t = (c.get("title", "") or "").split(" - ")[0].split(" | ")[0].strip()
        if not (t and 2 <= len(t.split()) <= 5) or t in detected_names:
            continue
        if name_hint and _name_sim(name_hint, t) >= 0.6:
            detected_names.append(t)
    if not detected_names and name_hint:
        detected_names = [name_hint]

    # Enrich confirmed social profiles with identity data
    for c in confirmed:
        if _platform(c["url"]) != "Web":
            _enrich(c)

    # CSE name-based expansion after face confirms a name
    cse_extra = []
    if detected_names:
        cse_extra = _google_cse(detected_names[0], None)
        for cr in cse_extra:
            if cr.get("photo_url"):
                _verify_one(query_np, cr)

    # Final merge
    final = confirmed + cse_extra + unconfirmed
    seen2, deduped = set(), []
    for m in final:
        u = m.get("url", "")
        if u and u not in seen2:
            seen2.add(u); deduped.append(m)

    n_confirmed = sum(1 for m in deduped if m.get("face_verified"))
    logger.info(
        f"reverse_face complete: {len(deduped)} matches, "
        f"{n_confirmed} face-confirmed, names={detected_names}"
    )

    return {
        "source":               "reverse_face",
        "matches":              deduped,
        "names_found":          detected_names,
        "face_confirmed_count": n_confirmed,
    }