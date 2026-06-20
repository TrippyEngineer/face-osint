"""
scrapers/username.py
─────────────────────────────────────────────────────────────────────
Three-layer username intelligence:

Layer 1 — Sherlock (300+ platforms via subprocess)
  pip install sherlock-project
  Calls: sherlock --print-found --no-color --csv --timeout 8 <variants>
  Parses CSV output → list of {platform, username, url}

Layer 2 — socid_extractor (100+ methods, extracts hidden IDs + cross-links)
  pip install socid-extractor
  For each found URL → requests.get → socid_extractor.extract(html)
  Surfaces: numeric IDs, linked accounts on OTHER platforms, emails

Layer 3 — Direct HEAD checks (25 hand-picked platforms, instant)
  Pure HTTP HEAD requests, no keys, no auth.
  Falls back gracefully if Sherlock not installed.

Name variant generation:
  "John Smith" → johnsmith, john.smith, john_smith,
                 jsmith, johnS, smith.john, john, smith …
"""
import sys
import ast
import csv
import json
import logging
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait
from typing import Optional

import requests
import config

logger = logging.getLogger(__name__)

# ── Name variant generator ────────────────────────────────────────────────
def _variants(full_name: str) -> list[str]:
    p     = full_name.lower().strip().split()
    if not p:
        return []
    first = p[0]
    last  = p[-1] if len(p) > 1 else ""
    fi    = first[0] if first else ""
    li    = last[0]  if last  else ""
    raw   = [
        f"{first}{last}",
        f"{first}.{last}",
        f"{first}_{last}",
        f"{fi}{last}",
        f"{first}{li}",
        f"{last}{first}",
        f"{last}.{first}",
        f"{last}_{first}",
        f"{fi}.{last}",
        f"{last}{fi}",
        f"{first}{last[:4]}",
        f"{first}",
        f"{last}",
        # Common numbering suffix handled by Sherlock's {?} syntax
    ]
    seen, out = set(), []
    for v in raw:
        v = v.strip("._").lower()
        if len(v) >= 2 and v not in seen:
            seen.add(v)
            out.append(v)
    # Top 5 only: more variants = O(n×325) HTTP requests and Sherlock timeouts.
    # The 5 most distinctive patterns cover >90% of real account naming conventions.
    return out[:5]


# ── Layer 1: Sherlock subprocess ──────────────────────────────────────────
def _sherlock(variants: list[str], timeout_s: int = 45) -> list[dict]:
    """
    Call sherlock CLI via subprocess. Returns list of found accounts.
    Gracefully returns [] if sherlock-project is not installed.
    """
    sherlock_cmd = shutil.which("sherlock")
    if not sherlock_cmd:
        # Try running as python module — guard the check itself: under high
        # concurrency the cold import of sherlock_project (300+ site defs) can
        # exceed the timeout, and an uncaught TimeoutExpired here would crash the
        # whole username scraper (observed once SCRAPER_MAX_WORKERS was raised).
        try:
            result = subprocess.run(
                [sys.executable, "-m", "sherlock_project.sherlock", "--version"],
                capture_output=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.info(f"sherlock-project version check failed ({type(e).__name__}) — skipping Layer 1")
            return []
        if result.returncode != 0:
            logger.info("sherlock-project not installed — skipping Layer 1")
            logger.info("  → pip install sherlock-project")
            return []
        sherlock_cmd = None   # use -m form below

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = (
            [sys.executable, "-m", "sherlock_project.sherlock"]
            if not sherlock_cmd
            else [sherlock_cmd]
        )
        cmd += [
            "--print-found",
            "--no-color",
            "--csv",
            "--timeout", "8",
            "--folderoutput", tmpdir,
        ] + variants

        logger.info(f"Sherlock: checking {len(variants)} variants on 300+ platforms")
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Sherlock timed out — partial results may exist")
        except Exception as e:
            logger.warning(f"Sherlock subprocess failed: {e}")
            return []

        # Parse all CSV files written to tmpdir
        matches = []
        seen_urls = set()
        for csv_file in Path(tmpdir).glob("*.csv"):
            try:
                with open(csv_file, encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        url = row.get("url", "").strip()
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            # Sherlock CSV: columns are username, site_name, url, ...
                            # site_name is the platform; csv_file.stem is the variant searched
                            platform = (
                                row.get("site_name")
                                or row.get("siteName")
                                or row.get("name")
                                or ""
                            ).strip()
                            matches.append({
                                "platform": platform,
                                "username": csv_file.stem,   # filename = variant searched
                                "url":      url,
                                "source":   "sherlock",
                            })
            except Exception as e:
                logger.debug(f"Sherlock CSV parse error {csv_file}: {e}")

        logger.info(f"Sherlock: {len(matches)} accounts found")
        return matches


# ── Layer 2: socid_extractor — hidden IDs + cross-platform links ──────────
def _socid_enrich(url: str) -> dict:
    """
    Fetch URL and extract all hidden identifiers.
    Returns socid dict or {}.
    Works on: GitHub, GitLab, Dev.to, Patreon, DeviantArt, npm, PyPI,
              Steam, HackerNews, Replit, Gravatar, Keybase, and 90+ more.
    """
    try:
        import socid_extractor
    except ImportError:
        return {}
    try:
        r = requests.get(
            url,
            headers=config.BROWSER_HEADERS,
            timeout=8,
        )
        data = socid_extractor.extract(r.text)
        return data or {}
    except Exception as e:
        logger.debug(f"socid_extractor failed for {url}: {e}")
        return {}


# ── Layer 3: Direct HEAD checks (25 platforms, instant, no keys) ──────────
# Columns: (platform, url_template, fail_status_code, fail_text_substring)
#
# fail_text MUST be set for any platform that returns HTTP 200 for non-existent
# users (they render their own "not found" page with a 200 status). Without it
# every variant produces a false positive.
#
# HEAD is used when fail_text is None (status code is enough, no body needed).
# GET  is used when fail_text is set  (we need to read the response body).
_DIRECT_SITES = [
    # Developer / professional — 404 reliable
    ("GitHub",     "https://github.com/{}",                       404,  None),
    ("GitLab",     "https://gitlab.com/{}",                       404,  None),
    ("Dev.to",     "https://dev.to/{}",                           404,  None),
    ("npm",        "https://www.npmjs.com/~{}",                    404,  None),
    ("PyPI",       "https://pypi.org/user/{}/",                    404,  None),
    ("Keybase",    "https://keybase.io/{}",                        404,  None),
    ("Replit",     "https://replit.com/@{}",                       404,  None),
    ("Pastebin",   "https://pastebin.com/u/{}",                    404,  None),
    ("Gravatar",   "https://en.gravatar.com/{}",                   404,  None),
    ("Linktree",   "https://linktr.ee/{}",                         404,  None),
    ("Flickr",     "https://www.flickr.com/people/{}",             404,  None),
    ("Mastodon",   "https://mastodon.social/@{}",                  404,  None),
    ("Codecademy", "https://www.codecademy.com/profiles/{}",       404,  None),
    ("Behance",    "https://www.behance.net/{}",                   404,  None),
    ("Dribbble",   "https://dribbble.com/{}",                      404,  None),
    ("Medium",     "https://medium.com/@{}",                       404,  None),
    # Text-check required — these return HTTP 200 for non-existent usernames
    ("HackerNews", "https://news.ycombinator.com/user?id={}",      None, "no such user"),
    ("Steam",      "https://steamcommunity.com/id/{}",             None, "the specified profile could not be found"),
    ("Instagram",  "https://www.instagram.com/{}/",                None, "sorry, this page isn't available"),
    ("TikTok",     "https://www.tiktok.com/@{}",                   None, "couldn't find this account"),
    ("YouTube",    "https://www.youtube.com/@{}",                  None, "this page isn't available"),
    ("Pinterest",  "https://www.pinterest.com/{}/",                None, "sorry! we couldn't find that page"),
    ("Spotify",    "https://open.spotify.com/user/{}",             None, "this content is not available"),
    ("Telegram",   "https://t.me/{}",                              None, "if you have telegram, you can contact"),
    ("Twitch",     "https://www.twitch.tv/{}",                     None, "sorry. unless you've got a time machine"),
]

def _check_one(platform: str, url_tpl: str, fail_code: Optional[int],
               fail_text: Optional[str], username: str) -> Optional[dict]:
    url = url_tpl.format(username)
    try:
        if fail_text:
            # Must download body to check the "not found" string
            r = requests.get(
                url,
                headers=config.BROWSER_HEADERS,
                timeout=6,
                allow_redirects=True,
            )
        else:
            # HEAD is enough — status code tells us everything
            r = requests.head(
                url,
                headers=config.BROWSER_HEADERS,
                timeout=6,
                allow_redirects=True,
            )

        if fail_code and r.status_code == fail_code:
            return None
        if fail_text and fail_text in r.text.lower():
            return None
        if r.status_code in (200, 301, 302):
            return {
                "platform": platform,
                "username": username,
                "url":      url,
                "source":   "direct_check",
            }
    except Exception:
        pass
    return None


def _direct_checks(variants: list[str]) -> list[dict]:
    logger.info(f"Direct checks: {len(variants)} × {len(_DIRECT_SITES)} = "
                f"{len(variants)*len(_DIRECT_SITES)} requests (max_workers=30)")
    found = []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futs = {
            pool.submit(_check_one, plat, tpl, fc, ft, uname): (plat, uname)
            for uname in variants
            for plat, tpl, fc, ft in _DIRECT_SITES
        }
        done_futs, _ = fut_wait(futs, timeout=20)
        for f in done_futs:
            try:
                r = f.result()
                if r:
                    found.append(r)
            except Exception:
                pass
    # Deduplicate by URL
    seen, deduped = set(), []
    for m in found:
        if m["url"] not in seen:
            seen.add(m["url"])
            deduped.append(m)
    logger.info(f"Direct checks: {len(deduped)} accounts found")
    return deduped


# ── Layer 4: Playwright — JS-gated profile verification + enrichment ─────
_JS_GATED_PLATFORMS = {
    "Instagram", "TikTok", "YouTube", "Pinterest", "Spotify", "Telegram", "Twitch"
}

_NOT_FOUND_PHRASES = [
    "this page isn't available",
    "sorry, this page",
    "couldn't find this account",
    "page not found",
    "user not found",
    "doesn't exist",
    "not available",
    "no results found",
    "profile not found",
]

def _playwright_verify_js_profiles(candidates: list[dict], deadline: float = 0.0) -> list[dict]:
    """
    Layer 4: Re-verify JS-gated platforms where plain HTTP lies.
    Instagram/TikTok/YouTube etc. return HTTP 200 with JS-rendered "not found" pages
    that requests.get can't see reliably. Playwright renders the real page.

    Drops confirmed false positives. Enriches confirmed hits with name/bio/avatar.
    Platforms not in _JS_GATED_PLATFORMS pass through unchanged.
    """
    js_hits = [m for m in candidates if m.get("platform") in _JS_GATED_PLATFORMS]
    if not js_hits:
        return candidates

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("playwright not installed — skipping JS profile verification")
        return candidates

    verified: dict[str, bool] = {}   # url → True=exists, False=not found
    enriched: dict[str, dict] = {}   # url → {name, bio, photo_url}

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(
                user_agent=config.BROWSER_HEADERS["User-Agent"],
                viewport={"width": 1280, "height": 900},
            )

            for m in js_hits[:10]:  # cap to stay within the scraper budget
                # PERF: stop launching page checks once over the total budget.
                # Each goto+networkidle below costs ~9s; verifying all candidates
                # is what ran username to 140s+. The orchestrator abandons us at
                # 50s regardless, so anything past the deadline is wasted work.
                if deadline and time.time() >= deadline:
                    logger.info("username Layer 4: stopping JS verify — over time budget")
                    break
                url      = m.get("url", "")
                platform = m.get("platform", "")
                if not url:
                    continue
                page = ctx.new_page()
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=6000)
                    page.wait_for_load_state("networkidle", timeout=3000)
                    content = page.content().lower()

                    exists = not any(p in content for p in _NOT_FOUND_PHRASES)
                    verified[url] = exists

                    if exists:
                        extra = {}
                        # Platform-specific data extraction from JSON blobs in <script> tags
                        if platform == "Instagram":
                            for script in page.locator('script[type="application/json"]').all():
                                try:
                                    raw = script.inner_text(timeout=400)
                                    if '"username"' not in raw:
                                        continue
                                    data = json.loads(raw)
                                    # Instagram buries profile data in nested dicts
                                    def _dig(obj, keys):
                                        for k in keys:
                                            if isinstance(obj, dict) and k in obj:
                                                return obj[k]
                                        if isinstance(obj, dict):
                                            for v in obj.values():
                                                r = _dig(v, keys)
                                                if r:
                                                    return r
                                        return ""
                                    extra["name"]      = _dig(data, ["full_name", "fullName"])
                                    extra["bio"]       = _dig(data, ["biography", "bio"])
                                    extra["photo_url"] = _dig(data, ["profile_pic_url_hd", "profilePicUrl", "profile_pic_url"])
                                    break
                                except Exception:
                                    continue

                        elif platform == "TikTok":
                            try:
                                raw = page.locator("#SIGI_STATE").inner_text(timeout=1000)
                                data = json.loads(raw)
                                users = data.get("UserModule", {}).get("users", {})
                                if users:
                                    u = next(iter(users.values()))
                                    extra["name"]      = u.get("nickname", "")
                                    extra["bio"]       = u.get("signature", "")
                                    extra["photo_url"] = u.get("avatarLarger", "")
                            except Exception:
                                pass

                        elif platform == "YouTube":
                            # og:title contains channel name
                            try:
                                t = page.locator('meta[property="og:title"]').get_attribute("content", timeout=800)
                                if t:
                                    extra["name"] = t.strip()
                            except Exception:
                                pass

                        if extra:
                            enriched[url] = extra

                except Exception as e:
                    logger.debug(f"Playwright verify {url}: {e}")
                    verified.setdefault(url, True)  # assume exists on error (don't drop)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

            browser.close()

    except Exception as e:
        logger.warning(f"Playwright profile verify: {e}")
        return candidates  # fallback: return original list unchanged

    # Rebuild candidate list — drop confirmed non-existent, enrich confirmed hits
    kept, dropped = [], 0
    for m in candidates:
        url = m.get("url", "")
        if url in verified:
            if not verified[url]:
                dropped += 1
                continue
            if url in enriched:
                for k, v in enriched[url].items():
                    if v:
                        m.setdefault(k, v)
            m["playwright_verified"] = True
        kept.append(m)

    logger.info(
        f"Playwright verify: {len(js_hits)} JS-gated profiles checked, "
        f"{dropped} false positives removed, {len(enriched)} enriched"
    )
    return kept


# ── Main scrape entry point ───────────────────────────────────────────────
def scrape(context: dict) -> dict:
    """
    Called by the SCRAPER dispatcher in app.py.
    Returns standard {source, matches} dict.
    """
    # PERF: total wall-clock budget. run_search gives this scraper a 50s deadline;
    # without an internal budget the layers below (socid enrichment, Playwright)
    # ran the thread to ~156s in the background. Run fast→slow and skip the slow
    # layers once the budget is spent so the scraper self-terminates near deadline.
    BUDGET_S  = 40   # under the 50s orchestrator deadline, leaving margin for one
                     # in-flight Layer-4 page load so the scraper self-terminates
    t0        = time.time()
    deadline  = t0 + BUDGET_S
    remaining = lambda: deadline - time.time()

    name       = context["name"]
    variants   = _variants(name)
    all_matches: list[dict] = []
    socid_data: list[dict]  = []

    logger.info(f"Username OSINT: name='{name}' → {len(variants)} variants: {variants}")

    # Layer 3 always runs (fast, no deps)
    direct = _direct_checks(variants)
    all_matches.extend(direct)

    # Layer 1: Sherlock — cap by remaining budget (subprocess is killed at timeout)
    if remaining() > 5:
        sherlock_results = _sherlock(variants, timeout_s=min(15, int(remaining())))
        # Merge Sherlock results, skip duplicates already found in direct
        seen_urls = {m["url"] for m in all_matches}
        for m in sherlock_results:
            if m["url"] not in seen_urls:
                seen_urls.add(m["url"])
                all_matches.append(m)
    else:
        logger.info("username: skipping Sherlock — over time budget")

    # Layer 2: socid_extractor enrichment on top 8 found URLs — stop at deadline
    enriched_extra: list[dict] = []
    for m in all_matches[:8]:
        if remaining() <= 2:
            logger.info("username: stopping socid enrichment — over time budget")
            break
        socid = _socid_enrich(m["url"])
        if socid:
            m["socid"] = socid
            # Extract any cross-platform links socid found
            links_raw = socid.get("links", "")
            if links_raw:
                try:
                    # ast.literal_eval handles Python-repr lists/dicts safely,
                    # unlike replace("'",'"') which breaks on apostrophes in values
                    links = ast.literal_eval(links_raw) if isinstance(links_raw, str) else links_raw
                    if isinstance(links, str):
                        links = [links]
                    for link in (links if isinstance(links, list) else []):
                        if isinstance(link, str) and link.startswith("http"):
                            enriched_extra.append({
                                "platform": "socid_link",
                                "url":      link,
                                "username": socid.get("username", ""),
                                "name":     socid.get("fullname", ""),
                                "source":   "socid_extractor",
                            })
                except Exception:
                    pass
            socid_data.append({"url": m["url"], "data": socid})

    # Add enriched links (deduplicated)
    seen_urls = {m["url"] for m in all_matches}
    for m in enriched_extra:
        if m["url"] not in seen_urls:
            seen_urls.add(m["url"])
            all_matches.append(m)

    # Layer 4: Playwright JS-profile verification — slowest layer; skip if budget spent
    if remaining() > 5:
        all_matches = _playwright_verify_js_profiles(all_matches, deadline=deadline)
    else:
        logger.info("username: skipping Layer 4 playwright verify — over time budget")

    logger.info(
        f"Username OSINT complete in {time.time()-t0:.1f}s: "
        f"{len(all_matches)} total accounts, "
        f"{len(socid_data)} socid enrichments, "
        f"variants={variants}"
    )

    return {
        "source":   "username",
        "matches":  all_matches,
        "variants": variants,
        "socid_enrichments": socid_data,
    }