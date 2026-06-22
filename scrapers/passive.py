"""
scrapers/passive.py  v2
─────────────────────────────────────────────────────────────────────
Zero auth. Zero scraping. All public APIs.

Sources:
  Wayback Machine   — archived URLs matching the name
  GDELT             — news article count mentioning the name
  crt.sh            — TLS certificate domain registration
  memory.lol        — historical Twitter/X username changes  
  Wayback Twitter   — mementoweb.org archived tweets         
  PGP keyservers    — public encryption keys (leak email)    
  Gravatar          — email hash → avatar (no email needed)  
  HaveIBeenPwned    — breach check by username  (no key needed for public endpoint)

All return empty gracefully on timeout — never crash the pipeline.
"""
import hashlib
import logging
import requests
import config

logger = logging.getLogger(__name__)


def scrape(context: dict) -> dict:
    name = context["name"]

    results = {
        "source":            "passive",
        "matches":           [],       # kept for scorer compat
        "wayback_urls":      [],
        "gdelt_mentions":    0,
        "crt_domains":       [],
        "twitter_history":   [],       # memory.lol results
        "wayback_tweets":    [],       # archived tweet URLs
        "pgp_keys":          [],       # PGP key entries
        "gravatar":          None,     # gravatar data if found
    }

    # Run all sub-scrapers, never let one crash the whole function
    for fn, key, label in [
        (_wayback,         "wayback_urls",    "Wayback Machine"),
        (_gdelt,           "gdelt_mentions",  "GDELT News"),
        (_pgp,             "pgp_keys",        "PGP keyservers"),
    ]:
        try:
            results[key] = fn(name)
            v = results[key]
            n = len(v) if isinstance(v, list) else v
            logger.debug(f"passive.{label}: {n}")
        except Exception as e:
            logger.warning(f"passive.{label} failed: {e}")

    # Username-dependent sources
    variants = _make_variants(name)
    for uname in variants[:4]:
        # memory.lol — historical Twitter names
        hist = _memory_lol(uname)
        if hist:
            results["twitter_history"].extend(hist)
            break

    # Wayback archived tweets — dropped: timetravel.mementoweb.org fails DNS and
    # stalls passive past its 20s deadline (wayback_tweets stays [] from init).

    # Gravatar (no email needed — try name variants as email prefixes)
    for uname in variants[:3]:
        grav = _gravatar(uname)
        if grav:
            results["gravatar"] = grav
            # Add as a match so face_matcher can compare the avatar
            results["matches"].append({
                "source":     "gravatar",
                "username":   uname,
                "url":        f"https://gravatar.com/{uname}",
                "photo_url":  grav.get("avatar_url"),
                "name":       grav.get("displayName"),
                "email_hint": f"{uname}@...",
            })
            break

    # Hunter.io email finder (only when API key is configured)
    hunter_key = getattr(config, "HUNTER_API_KEY", "") or ""
    if hunter_key:
        company = context.get("company", "")
        try:
            hunter_results = _hunter_email(name, company, hunter_key)
            results["matches"].extend(hunter_results)
        except Exception as e:
            logger.warning(f"passive.Hunter.io failed: {e}")

    # Email expansion — check context for email field and enrich matches
    context_email = context.get("email", "")
    if context_email and "@" in context_email:
        try:
            email_intel = _expand_email(context_email)
            if email_intel:
                results["matches"].append({
                    "source":  "email_expand",
                    "email":   context_email,
                    "url":     email_intel.get("domain_url", ""),
                    **email_intel,
                })
                logger.info(f"passive.email_expand: enriched {context_email}")
        except Exception as e:
            logger.debug(f"passive.email_expand: {e}")

    # Also expand any emails found in Hunter.io results
    for m in results["matches"]:
        found_email = m.get("email", "")
        if found_email and "@" in found_email and found_email != context_email:
            try:
                email_intel = _expand_email(found_email)
                if email_intel.get("gravatar_url") and not m.get("gravatar_url"):
                    m["gravatar_url"]     = email_intel["gravatar_url"]
                    m["gravatar_profile"] = email_intel.get("gravatar_profile", "")
                if email_intel.get("email_domain"):
                    m["email_domain"] = email_intel["email_domain"]
            except Exception:
                pass

    total = (
        len(results["wayback_urls"])
        + results["gdelt_mentions"]
        + len(results["crt_domains"])
        + len(results["twitter_history"])
        + len(results["pgp_keys"])
        + len(results["matches"])
    )
    logger.info(f"passive: {total} total intel items")
    return results


# ── Wayback Machine ───────────────────────────────────────────────────────
def _wayback(name: str) -> list:
    # Non-critical source — hard-capped at 8s so it never blocks the pipeline
    try:
        r = requests.get(
            "http://web.archive.org/cdx/search/cdx",
            params={
                "url":      f"*{name.replace(' ', '*')}*",
                "output":   "json",
                "limit":    15,
                "collapse": "urlkey",
                "fl":       "original,timestamp",
            },
            timeout=8,   # reduced from config.HTTP_TIMEOUT_S (12s) — non-critical
        )
        rows = r.json()
        return [f"https://web.archive.org/web/{row[1]}/{row[0]}"
                for row in rows[1:] if row]
    except Exception:
        return []


# ── GDELT News Count ──────────────────────────────────────────────────────
def _gdelt(name: str) -> int:
    # Non-critical source — hard-capped at 8s so it never blocks the pipeline
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": f'"{name}"', "mode": "artlist",
                    "maxrecords": 10, "format": "json"},
            timeout=8,   # reduced from config.HTTP_TIMEOUT_S (12s) — non-critical
        )
        return len(r.json().get("articles", []))
    except Exception:
        return 0


# ── crt.sh TLS certificates ───────────────────────────────────────────────
def _crt_sh(name: str) -> list:
    try:
        r = requests.get(
            "https://crt.sh/",
            params={"q": name, "output": "json"},
            timeout=config.HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        certs = r.json()
        domains = list({
            c.get("name_value", "").strip()
            for c in certs[:30]
            if c.get("name_value")
        })
        return [d for d in domains if d and not d.startswith("*")][:10]
    except Exception as e:
        logger.debug(f"crt.sh failed: {e}")
        return []


# ── memory.lol — historical Twitter/X usernames ──────────────────────────
def _memory_lol(username: str) -> list:
    """
    memory.lol stores a history of all Twitter username changes.
    Free API. Returns list of dicts: {username, date}.
    """
    try:
        r = requests.get(
            f"https://api.memory.lol/v1/tw/{username}",
            headers=config.BROWSER_HEADERS,
            timeout=6,
        )
        if r.status_code == 200:
            data = r.json()
            accounts = data.get("accounts", [])
            if accounts:
                history = []
                for acc in accounts[:1]:   # first match
                    for screen_name, dates in (acc.get("screen_names") or {}).items():
                        history.append({
                            "username": screen_name,
                            "dates":    dates if isinstance(dates, list) else [dates],
                        })
                logger.info(f"memory.lol: {len(history)} historical usernames for @{username}")
                return history
    except Exception as e:
        logger.debug(f"memory.lol failed for {username}: {e}")
    return []


# ── Wayback archived tweets ───────────────────────────────────────────────
def _wayback_twitter(name: str, variants: list) -> list:
    """
    Use mementoweb.org TimeTravel API to find archived Twitter/X pages.
    Also checks twitter.com/search and nitter archives.
    """
    urls_to_check = [
        f"https://twitter.com/search?q={name.replace(' ', '%20')}",
    ]
    for v in variants[:3]:
        urls_to_check.append(f"https://twitter.com/{v}")
        urls_to_check.append(f"https://x.com/{v}")

    found = []
    for url in urls_to_check[:4]:
        try:
            r = requests.get(
                "http://timetravel.mementoweb.org/api/json/",
                params={"url": url},
                timeout=6,
            )
            if r.status_code == 200:
                data = r.json()
                # mementos → closest
                mementos = data.get("mementos", {})
                closest  = mementos.get("closest", {})
                uri      = closest.get("uri", [])
                if uri:
                    found.append(uri[0] if isinstance(uri, list) else uri)
        except Exception as e:
            logger.debug(f"Wayback Twitter failed for {url}: {e}")

    logger.debug(f"Wayback Twitter: {len(found)} archived URLs")
    return found[:6]


# ── PGP keyservers — public key → real name + email ─────────────────────
def _pgp(name: str) -> list:
    """
    Search OpenPGP keyservers. Public keys often contain real name + email.
    Uses keys.openpgp.org (the modern GDPR-compliant server).
    """
    try:
        r = requests.get(
            "https://keys.openpgp.org/vks/v1/search",
            params={"q": name},
            headers={"Accept": "application/json"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            keys = []
            for cert in data.get("certificates", [])[:5]:
                for uid in cert.get("userids", []):
                    keys.append({
                        "fingerprint": cert.get("fingerprint", "")[:16],
                        "name":  uid.get("name",  ""),
                        "email": uid.get("email", ""),
                        "created": uid.get("created", ""),
                    })
            logger.info(f"PGP: {len(keys)} keys found for '{name}'")
            return keys
    except Exception as e:
        logger.debug(f"PGP keyserver failed: {e}")
    return []


# ── Gravatar ─────────────────────────────────────────────────────────────
def _gravatar(username: str) -> dict:
    """
    Query Gravatar JSON profile API.
    Gravatar URLs are based on MD5 of email, but the profile API accepts username directly.
    Returns profile dict with avatar_url, displayName, etc.
    """
    try:
        r = requests.get(
            f"https://www.gravatar.com/{username}.json",
            headers=config.BROWSER_HEADERS,
            timeout=5,
        )
        if r.status_code == 200:
            entry = r.json().get("entry", [{}])[0]
            avatar_hash = entry.get("hash", "")
            result = {
                "username":    username,
                "displayName": entry.get("displayName", ""),
                "avatar_url":  f"https://www.gravatar.com/avatar/{avatar_hash}?s=256&d=404",
                "profile_url": entry.get("profileUrl", ""),
                "about_me":    entry.get("aboutMe", ""),
                "emails":      [e["value"] for e in entry.get("emails", [])
                                if e.get("primary")],
                "accounts":    [
                    {"shortname": a.get("shortname"), "url": a.get("url")}
                    for a in entry.get("accounts", [])[:6]
                ],
            }
            if avatar_hash:
                logger.info(f"Gravatar: found profile for '{username}'")
                return result
    except Exception as e:
        logger.debug(f"Gravatar failed for {username}: {e}")
    return {}


# ── Hunter.io email finder ────────────────────────────────────────────────
def _hunter_email(name: str, company: str, api_key: str) -> list:
    """
    Use Hunter.io Email Finder API to locate a professional email address.
    Requires HUNTER_API_KEY in .env. Returns [] gracefully on any failure.
    """
    parts = name.strip().split()
    first = parts[0] if parts else ""
    last  = parts[-1] if len(parts) > 1 else ""
    if not first or not last or not company:
        return []
    try:
        r = requests.get(
            "https://api.hunter.io/v2/email-finder",
            params={
                "first_name": first,
                "last_name":  last,
                "company":    company,
                "api_key":    api_key,
            },
            timeout=8,
        )
        r.raise_for_status()
        data  = r.json().get("data", {})
        email = data.get("email")
        if not email:
            return []
        confidence = data.get("score", 0) / 100
        logger.info(f"Hunter.io: found email for '{name}' at '{company}' (confidence={confidence:.0%})")
        return [{
            "source":     "hunter",
            "email":      email,
            "name":       name,
            "url":        "https://hunter.io",
            "score":      confidence,
            "company":    company,
        }]
    except Exception as e:
        logger.debug(f"Hunter.io failed: {e}")
        return []


# ── Email expansion ───────────────────────────────────────────────────────
def _expand_email(email: str) -> dict:
    """Given an email, look up additional intel."""
    result = {}

    # 1. Gravatar by real email hash (only if we have actual email, not username hash)
    gh = hashlib.md5(email.lower().strip().encode()).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{gh}?d=404&s=256"
    try:
        r = requests.head(gravatar_url, timeout=4, allow_redirects=True)
        if r.status_code == 200:
            result["gravatar_url"]     = gravatar_url
            result["gravatar_profile"] = f"https://www.gravatar.com/{gh}"
    except Exception:
        pass

    # 2. Simple domain extraction
    domain = email.split("@")[-1] if "@" in email else ""
    if domain and domain not in ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com"):
        result["email_domain"] = domain
        result["domain_url"]   = f"https://{domain}"

    return result


# ── Helpers ───────────────────────────────────────────────────────────────
def _make_variants(name: str) -> list:
    p     = name.lower().strip().split()
    first = p[0] if p else ""
    last  = p[-1] if len(p) > 1 else ""
    fi    = first[0] if first else ""
    return [v for v in [
        f"{first}{last}", f"{first}.{last}", f"{fi}{last}", first, last
    ] if len(v) >= 2]