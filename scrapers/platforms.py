"""
scrapers/platforms.py  v2
─────────────────────────────────────────────────────────────────────
GitHub REST API  — profile + commit search + repo stats + org memberships
GitLab REST API  — public projects + contributions             ← NEW
npm              — package author search                       ← NEW
PyPI             — package maintainer search                   ← NEW
Reddit           — PRAW + public JSON fallback (unchanged)

socid_extractor is called on found profile pages in username.py.
This scraper focuses on extracting structured data from known platforms.
"""
import logging
import requests
import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  GITHUB
# ══════════════════════════════════════════════════════════════════════════
def scrape_github(context: dict) -> dict:
    name    = context["name"]
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    matches = []

    # ── 1. User search ────────────────────────────────────────────────
    try:
        r = requests.get(
            "https://api.github.com/search/users",
            params={"q": name, "per_page": 5, "sort": "best-match"},
            headers=headers, timeout=config.HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        for user in r.json().get("items", [])[:3]:
            login = user.get("login", "")
            profile = _gh_profile(login, headers)
            if profile:
                # ── 2. Recent commits search ──────────────────────────
                commits = _gh_commits(login, name, headers)
                # ── 3. Org memberships ────────────────────────────────
                orgs = _gh_orgs(login, headers)
                # ── 4. Top repos with stars ───────────────────────────
                repos = _gh_top_repos(login, headers)
                profile.update({
                    "commits":       commits,
                    "orgs":          orgs,
                    "top_repos":     repos,
                })
                matches.append(profile)
    except Exception as e:
        logger.warning(f"GitHub user search failed: {e}")
        return {"source": "github", "matches": [], "error": str(e)}

    # ── 5. Commit-email search — surfaces real email + extra accounts ─
    try:
        r = requests.get(
            "https://api.github.com/search/commits",
            params={"q": f"author:{name}", "per_page": 5},
            headers={**headers, "Accept": "application/vnd.github.cloak-preview+json"},
            timeout=config.HTTP_TIMEOUT_S,
        )
        if r.status_code == 200:
            for item in r.json().get("items", [])[:3]:
                author = item.get("commit", {}).get("author", {})
                email  = author.get("email", "")
                if email and not email.endswith("@users.noreply.github.com"):
                    # Check if we already have this user — guard against null author
                    author_login = (item.get("author") or {}).get("login", "")
                    existing = next(
                        (m for m in matches
                         if author_login and m.get("username") == author_login),
                        None,
                    )
                    if existing:
                        existing["email"] = existing.get("email") or email
                    else:
                        matches.append({
                            "username":   (item.get("author") or {}).get("login", ""),
                            "email":      email,
                            "url":        item.get("html_url", ""),
                            "source_hint": "commit_search",
                        })
    except Exception as e:
        logger.debug(f"GitHub commit search: {e}")

    logger.info(f"GitHub: {len(matches)} profiles")
    return {"source": "github", "matches": matches}


def _gh_profile(login: str, headers: dict) -> dict:
    try:
        r  = requests.get(f"https://api.github.com/users/{login}",
                          headers=headers, timeout=config.HTTP_TIMEOUT_S)
        p  = r.json()
        return {
            "username":   p.get("login"),
            "name":       p.get("name"),
            "company":    (p.get("company") or "").strip("@"),
            "location":   p.get("location"),
            "email":      p.get("email"),
            "bio":        (p.get("bio") or "")[:140],
            "repos":      p.get("public_repos"),
            "followers":  p.get("followers"),
            "photo_url":  p.get("avatar_url"),
            "url":        p.get("html_url"),
            "blog":       p.get("blog"),
            "twitter":    p.get("twitter_username"),
            "created":    (p.get("created_at") or "")[:10],
        }
    except Exception:
        return {}


def _gh_commits(login: str, name: str, headers: dict) -> list:
    """Search recent commits — reveals coding activity and repos."""
    try:
        r = requests.get(
            "https://api.github.com/search/commits",
            params={"q": f"author:{login}", "per_page": 5,
                    "sort": "author-date", "order": "desc"},
            headers={**headers, "Accept": "application/vnd.github.cloak-preview+json"},
            timeout=8,
        )
        if r.status_code == 200:
            return [
                {"repo":    item.get("repository", {}).get("full_name", ""),
                 "message": (item.get("commit", {}).get("message") or "")[:80],
                 "date":    (item.get("commit", {}).get("author", {}).get("date") or "")[:10]}
                for item in r.json().get("items", [])[:5]
            ]
    except Exception:
        pass
    return []


def _gh_orgs(login: str, headers: dict) -> list:
    """Public org memberships — reveals employer/team affiliations."""
    try:
        r = requests.get(
            f"https://api.github.com/users/{login}/orgs",
            headers=headers, timeout=6,
        )
        if r.status_code == 200:
            return [{"name": o.get("login"), "url": f"https://github.com/{o.get('login')}"}
                    for o in r.json()[:6]]
    except Exception:
        pass
    return []


def _gh_top_repos(login: str, headers: dict) -> list:
    """Top repos by stars."""
    try:
        r = requests.get(
            f"https://api.github.com/users/{login}/repos",
            params={"sort": "stars", "per_page": 5},
            headers=headers, timeout=6,
        )
        if r.status_code == 200:
            return [
                {"name":    repo.get("name"),
                 "stars":   repo.get("stargazers_count", 0),
                 "lang":    repo.get("language"),
                 "url":     repo.get("html_url")}
                for repo in r.json()[:5]
                if repo.get("stargazers_count", 0) > 0
            ]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════
#  GITLAB  (new)
# ══════════════════════════════════════════════════════════════════════════
def scrape_gitlab(context: dict) -> dict:
    name    = context["name"]
    matches = []

    # GitLab requires auth for user search since 2024.
    # Skip gracefully if no token — avoids 403 spam in logs.
    gitlab_token = getattr(config, "GITLAB_TOKEN", "") or ""
    if not gitlab_token:
        logger.info("GitLab: no GITLAB_TOKEN in .env — skipping (add token to enable)")
        return {"source": "gitlab", "matches": [], "error": "No GITLAB_TOKEN"}

    gl_headers = {**config.BROWSER_HEADERS, "PRIVATE-TOKEN": gitlab_token}

    try:
        r = requests.get(
            "https://gitlab.com/api/v4/users",
            params={"search": name, "per_page": 5},
            headers=gl_headers,
            timeout=config.HTTP_TIMEOUT_S,
        )
        if r.status_code == 403:
            logger.warning("GitLab 403 — token may lack read_user scope")
            return {"source": "gitlab", "matches": [], "error": "403 Forbidden"}
        r.raise_for_status()
        for user in r.json()[:3]:
            uid  = user.get("id")
            # Fetch contributed projects
            proj = _gitlab_projects(uid, token=gitlab_token)
            matches.append({
                "username":   user.get("username"),
                "name":       user.get("name"),
                "url":        user.get("web_url"),
                "photo_url":  user.get("avatar_url"),
                "bio":        (user.get("bio") or "")[:140],
                "location":   user.get("location"),
                "website":    user.get("website_url"),
                "projects":   proj,
                "source":     "gitlab",
            })
    except Exception as e:
        logger.warning(f"GitLab search failed: {e}")
        return {"source": "gitlab", "matches": [], "error": str(e)}
    logger.info(f"GitLab: {len(matches)} profiles")
    return {"source": "gitlab", "matches": matches}


def _gitlab_projects(uid: int, token: str = "") -> list:
    try:
        hdrs = {**config.BROWSER_HEADERS}
        if token:
            hdrs["PRIVATE-TOKEN"] = token
        r = requests.get(
            f"https://gitlab.com/api/v4/users/{uid}/projects",
            params={"per_page": 5, "order_by": "star_count"},
            headers=hdrs, timeout=6,
        )
        if r.status_code == 200:
            return [{"name": p.get("name"), "stars": p.get("star_count", 0),
                     "url": p.get("web_url")} for p in r.json()[:5]]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════
#  npm package author search (new)
# ══════════════════════════════════════════════════════════════════════════
def scrape_npm(context: dict) -> dict:
    """
    npm registry: search packages by author name.
    No key needed. Returns published packages + author email if public.
    """
    name    = context["name"]
    matches = []
    try:
        r = requests.get(
            "https://registry.npmjs.org/-/v1/search",
            params={"text": f"author:{name.replace(' ', '+')}",
                    "size": 8},
            headers=config.BROWSER_HEADERS,
            timeout=config.HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        for obj in r.json().get("objects", [])[:5]:
            pkg     = obj.get("package", {})
            author  = pkg.get("author", {})
            matches.append({
                "platform":  "npm",
                "username":  author.get("username", ""),
                "name":      author.get("name", ""),
                "email":     author.get("email", ""),
                "pkg_name":  pkg.get("name"),
                "pkg_desc":  (pkg.get("description") or "")[:100],
                "url":       f"https://www.npmjs.com/~{author.get('username', '')}",
                "pkg_url":   f"https://www.npmjs.com/package/{pkg.get('name','')}",
                "source":    "npm",
            })
    except Exception as e:
        logger.warning(f"npm search failed: {e}")
        return {"source": "npm", "matches": [], "error": str(e)}
    logger.info(f"npm: {len(matches)} packages")
    return {"source": "npm", "matches": matches}


# ══════════════════════════════════════════════════════════════════════════
#  PyPI package maintainer search (new)
# ══════════════════════════════════════════════════════════════════════════
def scrape_pypi(context: dict) -> dict:
    """
    PyPI: search packages. No official author-search endpoint,
    but we can check the user profile page and search by name.
    """
    name    = context["name"]
    matches = []

    # Try username variants as direct PyPI profile lookups
    p       = name.lower().strip().split()
    first   = p[0] if p else ""
    last    = p[-1] if len(p) > 1 else ""
    for uname in [f"{first}{last}", f"{first}.{last}", f"{first}", f"{first}_{last}"]:
        if len(uname) < 2:
            continue
        try:
            r = requests.get(
                f"https://pypi.org/user/{uname}/",
                headers=config.BROWSER_HEADERS,
                timeout=6,
                allow_redirects=True,
            )
            if r.status_code == 200 and "pypi.org/user/" in r.url:
                from bs4 import BeautifulSoup
                soup     = BeautifulSoup(r.text, "html.parser")
                packages = [a.get_text(strip=True)
                            for a in soup.select(".package-snippet__name")[:10]]
                matches.append({
                    "platform": "PyPI",
                    "username": uname,
                    "url":      f"https://pypi.org/user/{uname}/",
                    "packages": packages,
                    "source":   "pypi",
                })
                break
        except Exception:
            pass

    logger.info(f"PyPI: {len(matches)} profiles")
    return {"source": "pypi", "matches": matches}


# ══════════════════════════════════════════════════════════════════════════
#  REDDIT (unchanged from v1 — kept here for import compat)
# ══════════════════════════════════════════════════════════════════════════
def scrape_reddit(context: dict) -> dict:
    name    = context["name"]
    matches = []

    if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            for sub in reddit.subreddit("all").search(name, limit=5):
                try:
                    if not sub.author:
                        continue
                    u        = reddit.redditor(sub.author.name)
                    top_subs = list({c.subreddit.display_name
                                     for c in u.comments.new(limit=25)})[:8]
                    matches.append({
                        "username":       u.name,
                        "karma":          u.link_karma + u.comment_karma,
                        "top_subreddits": top_subs,
                        "url":            f"https://reddit.com/u/{u.name}",
                    })
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"PRAW failed: {e}")

    if not matches:
        try:
            r = requests.get(
                "https://www.reddit.com/search.json",
                params={"q": name, "type": "user", "limit": 5},
                headers={"User-Agent": config.REDDIT_USER_AGENT},
                timeout=config.HTTP_TIMEOUT_S,
            )
            for child in r.json().get("data", {}).get("children", [])[:3]:
                d = child.get("data", {})
                matches.append({
                    "username": d.get("name"),
                    "karma":    d.get("total_karma", 0),
                    "url":      f"https://reddit.com/u/{d.get('name', '')}",
                })
        except Exception as e:
            logger.debug(f"Reddit public JSON failed: {e}")

    logger.info(f"Reddit: {len(matches)} matches")
    return {"source": "reddit", "matches": matches}