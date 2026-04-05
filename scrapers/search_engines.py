import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# Platform search sites — used for targeted dorks
_PLATFORM_SITES = {
    "linkedin":     "site:linkedin.com/in",
    "github":       "site:github.com",
    "twitter":      'site:twitter.com OR site:x.com',
    "instagram":    "site:instagram.com",
    "facebook":     "site:facebook.com",
    "researchgate": "site:researchgate.net",
    "medium":       "site:medium.com",
    "reddit":       "site:reddit.com/user",
    "behance":      "site:behance.net",
    "stackoverflow":"site:stackoverflow.com/users",
    "devto":        "site:dev.to",
    "youtube":      "site:youtube.com/@",
}


# ══════════════════════════════════════════════════════════════════════════
#  MAIN SCRAPER
# ══════════════════════════════════════════════════════════════════════════
def scrape(context: dict) -> dict:
    name     = context.get("name", "")
    company  = context.get("company", "")
    location = context.get("location", "")

    if not name:
        return {"source": "search_engines", "matches": []}

    logger.info(f"search_engines: searching '{name}' company='{company}' loc='{location}'")

    all_matches = []

    # ── 1. LinkedIn — highest priority, dedicated function ──────────────
    all_matches.extend(_search_linkedin(name, company, location))

    # ── 2. All other platforms in parallel ──────────────────────────────
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_search_platform, name, platform, site_dork): platform
            for platform, site_dork in _PLATFORM_SITES.items()
            if platform != "linkedin"   # already done above
        }
        for fut in as_completed(futures):
            try:
                all_matches.extend(fut.result())
            except Exception as e:
                logger.debug(f"platform search: {e}")

    # ── 3. General web search (news, PDFs, contacts) ─────────────────────
    all_matches.extend(_search_general(name, company))

    # ── 4. Google News RSS ───────────────────────────────────────────────
    all_matches.extend(_search_google_news(name, company, location))

    # ── 5. Instagram OSINT via DDG ──────────────────────────────────────
    all_matches.extend(_search_instagram(name, company, location))

    # Deduplicate by URL
    seen, deduped = set(), []
    for m in all_matches:
        u = m.get("url","")
        if u and u not in seen:
            seen.add(u); deduped.append(m)

    # Tag platform and add source_confidence heuristic
    _MAJOR_PLATFORMS = {
        "twitter.com", "x.com", "instagram.com", "github.com",
        "facebook.com", "youtube.com", "tiktok.com", "reddit.com",
    }
    for m in deduped:
        url = m.get("url","")
        if "linkedin.com" in url:
            m["platform"] = "linkedin"; m["is_linkedin"] = True
        elif "github.com" in url:
            m["platform"] = "github"
        elif "twitter.com" in url or "x.com" in url:
            m["platform"] = "twitter"
        elif "instagram.com" in url:
            m["platform"] = "instagram"
        elif "researchgate.net" in url:
            m["platform"] = "researchgate"

        # Source confidence heuristic
        m["source_confidence"] = _source_confidence(m, name, company, location)

    n_li = sum(1 for m in deduped if m.get("is_linkedin"))
    logger.info(f"search_engines: {len(deduped)} URLs total (linkedin: {n_li})")
    return {"source": "search_engines", "matches": deduped}


def _source_confidence(match: dict, name: str, company: str, location: str) -> float:
    """Heuristic confidence [0.0–1.0] for a search_engines match."""
    url     = match.get("url", "").lower()
    title   = (match.get("title", "") or "").lower()
    snippet = (match.get("snippet", "") or match.get("bio", "") or "").lower()

    if "linkedin.com/in/" in url and name.lower() in title:
        base = 0.7
    elif "linkedin.com" in url:
        base = 0.55
    else:
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc.lstrip("www.")
            base = 0.5 if any(p in host for p in _MAJOR_PLATFORMS) else 0.2
        except Exception:
            base = 0.2

    bonus = 0.0
    context_text = snippet + title
    if company and company.lower() in context_text:
        bonus += 0.1
    if location and location.lower() in context_text:
        bonus += 0.1

    return round(min(1.0, base + bonus), 3)


# ══════════════════════════════════════════════════════════════════════════
#  LINKEDIN — dedicated, tries multiple engines
# ══════════════════════════════════════════════════════════════════════════
def _search_linkedin(name: str, company: str = "", location: str = "") -> list[dict]:
    """
    Search LinkedIn profiles via Google/DDG (no LinkedIn account needed).
    Tries: Google CSE → DDG HTML → Bing HTML
    Returns profile URLs with headlines extracted from snippets.
    """
    queries = [f'"{name}" site:linkedin.com/in']
    if company:
        queries.append(f'"{name}" site:linkedin.com/in "{company}"')
    if location:
        queries.append(f'"{name}" site:linkedin.com/in "{location}"')

    results = []
    seen    = set()

    for q in queries:
        items = (_google_cse(q) or _ddg_html(q) or _bing_html(q))
        for item in items:
            url = item.get("url","")
            if "linkedin.com/in/" in url and url not in seen:
                seen.add(url)
                item["is_linkedin"] = True
                item["platform"]    = "linkedin"
                # Extract name + headline from snippet
                snippet = item.get("snippet","")
                _parse_linkedin_snippet(item, snippet)
                results.append(item)

    if results:
        logger.info(f"LinkedIn: {len(results)} profiles found for '{name}'")
        _fetch_linkedin_public(results)
    else:
        logger.warning(f"LinkedIn: 0 profiles found for '{name}' — "
                       f"(set GOOGLE_CSE_KEY in .env for best results)")
    return results


def _parse_linkedin_snippet(match: dict, snippet: str):
    """Extract structured data from Google/DDG snippet of a LinkedIn result."""
    if not snippet:
        return
    # Snippet often looks like: "Name · Title at Company · Location"
    # or "Name - Software Engineer at Google - Bangalore, India"
    match["bio"] = snippet[:200]
    # Try to extract title/company
    for sep in (" · ", " - ", " | "):
        parts = snippet.split(sep)
        if len(parts) >= 2:
            match.setdefault("title", parts[0].strip()[:100])
            if len(parts) >= 3:
                match.setdefault("company_hint", parts[1].strip()[:80])
            break
    # Extract username from URL
    url = match.get("url","")
    m = re.search(r"linkedin\.com/in/([^/?#]+)", url)
    if m:
        match["linkedin_username"] = m.group(1)


def _fetch_linkedin_public(profiles: list) -> None:
    """
    Attempt to fetch up to 3 LinkedIn public profile pages (no auth) and
    enrich each match dict in-place with linkedin_headline and
    linkedin_location parsed from the public page.
    Wraps all errors — never raises.
    """
    fetched = 0
    for match in profiles:
        if fetched >= 3:
            break
        url = match.get("url", "")
        if "linkedin.com/in/" not in url:
            continue
        try:
            r = requests.get(
                url,
                headers={
                    **config.BROWSER_HEADERS,
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=8,
                allow_redirects=True,
            )
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Headline — several selectors tried in order
            headline = ""
            for sel in (
                "h2.top-card-layout__headline",
                ".top-card-layout__headline",
                '[class*="headline"]',
                "meta[name='description']",
            ):
                el = soup.select_one(sel)
                if el:
                    headline = (el.get("content") or el.get_text()).strip()[:200]
                    if headline:
                        break
            # Fallback: og:description often contains "Title at Company"
            if not headline:
                og = soup.find("meta", property="og:description")
                if og:
                    headline = (og.get("content") or "").strip()[:200]

            # Location
            location = ""
            for sel in (
                ".top-card__subline-item",
                '[class*="location"]',
                "meta[name='keywords']",
            ):
                el = soup.select_one(sel)
                if el:
                    location = (el.get("content") or el.get_text()).strip()[:120]
                    if location:
                        break

            if headline:
                match["linkedin_headline"] = headline
            if location:
                match["linkedin_location"] = location

            fetched += 1
            logger.debug(f"LinkedIn public fetch OK: {url[:80]}")
        except Exception as e:
            logger.debug(f"LinkedIn public fetch failed for {url[:80]}: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  PLATFORM SEARCH — single platform query
# ══════════════════════════════════════════════════════════════════════════
def _search_platform(name: str, platform: str, site_dork: str) -> list[dict]:
    q = f'"{name}" {site_dork}'
    items = _google_cse(q) or _ddg_html(q) or _bing_html(q)
    for item in items:
        item["platform"] = platform
    return items[:5]


# ══════════════════════════════════════════════════════════════════════════
#  GENERAL WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════
def _search_general(name: str, company: str = "") -> list[dict]:
    queries = [f'"{name}" {company}'.strip()]
    if company:
        queries.append(f'"{name}" "{company}" contact OR profile OR biography')

    results = []
    seen    = set()
    for q in queries[:2]:
        for item in (_google_cse(q) or _ddg_html(q) or _bing_html(q)):
            url = item.get("url","")
            if url and url not in seen:
                seen.add(url); results.append(item)
    return results[:10]


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH ENGINES — each returns list[dict] with url/title/snippet/source
# ══════════════════════════════════════════════════════════════════════════
def _google_cse(query: str) -> list:
    if not getattr(config, "GOOGLE_CSE_KEY","") or not getattr(config, "GOOGLE_CSE_ID",""):
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": config.GOOGLE_CSE_KEY, "cx": config.GOOGLE_CSE_ID,
                    "q": query, "num": 10},
            timeout=config.HTTP_TIMEOUT_S,
        )
        items = r.json().get("items", [])
        results = [{"url": i["link"], "snippet": i.get("snippet","")[:300],
                    "title": i.get("title",""), "source": "google_cse"}
                   for i in items if i.get("link")]
        if results:
            logger.debug(f"Google CSE '{query[:60]}': {len(results)}")
        return results
    except Exception as e:
        logger.debug(f"Google CSE: {e}")
        return []


def _ddg_html(query: str) -> list:
    """DuckDuckGo HTML — no key, reliable, respects site: operators."""
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={**config.BROWSER_HEADERS, "Accept": "text/html"},
            timeout=12,
        )
        soup    = BeautifulSoup(r.text, "html.parser")
        results = []
        for div in soup.select(".result__body, .result")[:10]:
            a       = div.select_one(".result__title a, a.result__a")
            snippet = div.select_one(".result__snippet")
            if not a: continue
            href = a.get("href","")
            if "uddg=" in href:
                qs   = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = urllib.parse.unquote(qs.get("uddg",[""])[0])
            if href.startswith("http"):
                results.append({
                    "url":     href,
                    "title":   a.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True)[:300] if snippet else "",
                    "source":  "ddg_html",
                })
        if results:
            logger.debug(f"DDG HTML '{query[:60]}': {len(results)}")
        return results
    except Exception as e:
        logger.debug(f"DDG HTML: {e}")
        return []


def _bing_html(query: str) -> list:
    """Bing HTML scraping — no key needed."""
    try:
        r = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": 10, "setlang": "en-US"},
            headers={**config.BROWSER_HEADERS,
                     "Accept": "text/html,application/xhtml+xml"},
            timeout=10,
        )
        soup    = BeautifulSoup(r.text, "html.parser")
        results = []
        for li in soup.select(".b_algo")[:8]:
            a       = li.select_one("h2 a")
            snippet = li.select_one(".b_caption p, .b_algoSlug, .b_dList")
            if a and a.get("href","").startswith("http"):
                results.append({
                    "url":     a["href"],
                    "title":   a.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True)[:300] if snippet else "",
                    "source":  "bing_html",
                })
        if results:
            logger.debug(f"Bing HTML '{query[:60]}': {len(results)}")
        return results
    except Exception as e:
        logger.debug(f"Bing HTML: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
#  GOOGLE NEWS RSS
# ══════════════════════════════════════════════════════════════════════════
def _search_google_news(name: str, company: str = "", location: str = "") -> list:
    """
    Query the Google News RSS feed for mentions of the person.
    No API key needed. Returns up to 5 news items.
    """
    query = f'"{name}"'
    if company:
        query += f' "{company}"'
    if location:
        query += f' "{location}"'
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=config.BROWSER_HEADERS,
            timeout=8,
        )
        if not r.ok:
            return []
        root = ET.fromstring(r.content)
        ns   = {"media": "http://search.yahoo.com/mrss/"}
        results = []
        for item in root.findall(".//item")[:5]:
            link    = (item.findtext("link") or "").strip()
            title   = (item.findtext("title") or "").strip()
            pub     = (item.findtext("pubDate") or "").strip()
            desc    = (item.findtext("description") or "").strip()
            if not link:
                continue
            results.append({
                "source":  "google_news",
                "url":     link,
                "snippet": f"{title} — {pub}"[:300],
                "title":   title,
                "bio":     desc[:200],
                "name":    name,
            })
        if results:
            logger.info(f"Google News: {len(results)} articles for '{name}'")
        return results
    except Exception as e:
        logger.debug(f"Google News: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
#  INSTAGRAM OSINT (via DuckDuckGo site: search — no login needed)
# ══════════════════════════════════════════════════════════════════════════
def _search_instagram(name: str, company: str = "", location: str = "") -> list:
    """
    Find Instagram profiles by searching DDG for site:instagram.com.
    No Instagram account or API key needed.
    """
    query = f'site:instagram.com "{name}"'
    if company:
        query += f' "{company}"'
    try:
        raw = _ddg_html(query)
        matches = []
        for r in raw[:5]:
            url = r.get("url", "")
            if "instagram.com/" in url:
                matches.append({
                    "source":  "instagram",
                    "url":     url,
                    "snippet": r.get("snippet", "")[:300],
                    "title":   r.get("title", ""),
                    "name":    name,
                    "platform": "instagram",
                })
        if matches:
            logger.info(f"Instagram (DDG): {len(matches)} profiles for '{name}'")
        return matches
    except Exception as e:
        logger.debug(f"Instagram DDG search: {e}")
        return []