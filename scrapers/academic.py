"""
scrapers/academic.py
Semantic Scholar + OpenAlex + ORCID
All three: completely free, no API key, no rate limit issues.
"""
import logging
import time
import requests
import config

logger = logging.getLogger(__name__)

# ── Simple in-memory 1hr cache for Semantic Scholar (rate-limited at 100/5min) ──
_ss_cache: dict = {}   # name_lower → (timestamp, results)
_SS_CACHE_TTL = 3600   # seconds


def scrape(context: dict) -> dict:
    name    = (context.get("name") or "").strip()
    matches = []

    # Without a name these APIs return garbage: OpenAlex yields the globally
    # largest authors for search='', Semantic Scholar 429s on query='', and
    # _orcid would IndexError on an empty split. A photo-only search has no name
    # signal, so there is nothing meaningful for academic sources to return.
    if not name:
        return {"source": "academic", "matches": []}

    for fn in (_google_scholar, _semantic_scholar, _open_alex, _orcid):
        try:
            results = fn(name)
            matches.extend(results)
            logger.debug(f"academic.{fn.__name__}: {len(results)} results")
        except Exception as e:
            logger.warning(f"academic.{fn.__name__} failed: {e}")
        # Semantic Scholar has a strict 100 req/5 min rate limit; the in-memory
        # cache handles repeat queries. A short delay only after that call is enough.
        if fn is _semantic_scholar:
            time.sleep(0.5)

    return {"source": "academic", "matches": matches}


def _semantic_scholar(name: str) -> list:
    # Check cache first — avoids 429 when the same name is searched again within 1hr
    key = name.lower().strip()
    cached = _ss_cache.get(key)
    if cached:
        ts, results = cached
        if time.time() - ts < _SS_CACHE_TTL:
            logger.debug(f"Semantic Scholar: cache hit for '{name}'")
            return results

    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params={"query": name, "fields": "name,affiliations,paperCount,hIndex,url", "limit": 5},
        timeout=config.HTTP_TIMEOUT_S,
    )
    r.raise_for_status()
    results = [{
        "source":      "semantic_scholar",
        "name":        a.get("name"),
        "affiliation": (a.get("affiliations") or [{}])[0].get("name"),
        "paper_count": a.get("paperCount"),
        "h_index":     a.get("hIndex"),
        "profile_url": a.get("url"),
    } for a in r.json().get("data", [])[:3]]

    _ss_cache[key] = (time.time(), results)
    return results


def _open_alex(name: str) -> list:
    # mailto param puts the request in OpenAlex "polite pool" (higher rate limits).
    # Uses OPENALEX_MAILTO from .env if set, otherwise omits it.
    params: dict = {"search": name, "per-page": 5}
    openalex_email = getattr(config, "OPENALEX_MAILTO", "")
    if openalex_email:
        params["mailto"] = openalex_email
    r = requests.get(
        "https://api.openalex.org/authors",
        params=params,
        timeout=config.HTTP_TIMEOUT_S,
    )
    r.raise_for_status()
    return [{
        "source":      "openalex",
        "name":        a.get("display_name"),
        "affiliation": (a.get("last_known_institution") or {}).get("display_name"),
        "paper_count": a.get("works_count"),
        "h_index":     a.get("summary_stats", {}).get("h_index"),
        "profile_url": a.get("id"),
    } for a in r.json().get("results", [])[:3]]


def _orcid(name: str) -> list:
    parts = name.strip().split()
    if not parts:                       # guard: empty name → no family-name token
        return []
    q     = f'family-name:{parts[-1]}'
    if len(parts) > 1:
        q += f' given-names:{parts[0]}'
    r = requests.get(
        "https://pub.orcid.org/v3.0/search/",
        params={"q": q},
        headers={"Accept": "application/json"},
        timeout=config.HTTP_TIMEOUT_S,
    )
    r.raise_for_status()
    results = r.json().get("result", [])
    return [{
        "source":      "orcid",
        "name":        name,
        "profile_url": f"https://orcid.org/{(res.get('orcid-identifier') or {}).get('path', '')}",
    } for res in results[:3]]


def _google_scholar(name: str) -> list:
    """
    Scrape Google Scholar profiles via DDG site search.
    No API key needed. Finds researchers with publications.
    """
    import re
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f'"{name}" site:scholar.google.com'},
            headers={**config.BROWSER_HEADERS, "Accept": "text/html"},
            timeout=config.HTTP_TIMEOUT_S,
        )
        from bs4 import BeautifulSoup
        import urllib.parse
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        seen_users: set = set()   # deduplicate by Scholar user ID
        for div in soup.select(".result__body, .result")[:8]:
            a = div.select_one(".result__title a, a.result__a")
            snip = div.select_one(".result__snippet")
            if not a:
                continue
            href = a.get("href", "")
            if "uddg=" in href:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = urllib.parse.unquote(qs.get("uddg", [""])[0])
            if "scholar.google" in href and "citations" in href:
                # Normalise to the en-language version; strip hl= param so
                # the same profile in different languages isn't added twice.
                parsed = urllib.parse.urlparse(href)
                qs2    = urllib.parse.parse_qs(parsed.query)
                user   = qs2.get("user", [""])[0]
                if not user or user in seen_users:
                    continue
                seen_users.add(user)
                norm_href = f"https://scholar.google.com/citations?user={user}&hl=en"
                results.append({
                    "source":      "google_scholar",
                    "name":        a.get_text(strip=True).split(" - ")[0],
                    "affiliation": snip.get_text(strip=True)[:100] if snip else "",
                    "profile_url": norm_href,
                    "url":         norm_href,
                })
        logger.info(f"Google Scholar (DDG): {len(results)} profiles for '{name}'")
        return results
    except Exception as e:
        logger.warning(f"Google Scholar DDG: {e}")
        return []