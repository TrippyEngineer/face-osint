"""
diagnose.py — Run this FIRST to see what's working
════════════════════════════════════════════════════
    python diagnose.py

Checks every component and tells you exactly what to fix.
"""
import sys, os

print("\n" + "═"*60)
print("  Face OSINT — Diagnostic Check")
print("═"*60)

# ── Python version ────────────────────────────────────────────────
print("\n[SYSTEM]")
v = sys.version_info
if v >= (3, 12):
    print(f"  WARN  Python {v.major}.{v.minor}.{v.micro} — TensorFlow 2.16 requires Python 3.10 or 3.11 ⚠️")
    print(f"        Python 3.12+ will break TensorFlow 2.16. Downgrade to 3.11 recommended.")
elif v < (3, 10):
    print(f"  WARN  Python {v.major}.{v.minor}.{v.micro} — minimum required is Python 3.10 ⚠️")
else:
    print(f"  OK    Python {v.major}.{v.minor}.{v.micro} ✅")

# ── .env keys ─────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(".env", override=False)

print("\n[API Keys in .env]")
keys = {
    "GOOGLE_CSE_KEY":  ("Google CSE — social text search",         "CRITICAL"),
    "GOOGLE_CSE_ID":   ("Google CSE ID",                           "CRITICAL"),
    "GITHUB_TOKEN":    ("GitHub API — 5000/hr vs 60/hr",           "recommended"),
    "GITLAB_TOKEN":    ("GitLab API — required since 2024",        "recommended"),
    "SERPAPI_KEY":     ("SerpApi — Google Lens + Yandex reverse",  "recommended"),
    "IMGBB_API_KEY":   ("imgbb — image hosting for SerpApi",       "recommended"),
    "BING_API_KEY":    ("Bing API",                                 "optional"),
    "BRAVE_API_KEY":   ("Brave Search",                            "optional"),
}
for k, (desc, priority) in keys.items():
    val = os.getenv(k,"")
    status = "✅ set" if val else f"❌ missing [{priority}]"
    print(f"  {k:<22} {status}   → {desc}")

# Optional keys — informational only
openalex_mail = os.getenv("OPENALEX_MAILTO", "")
if openalex_mail:
    print(f"  {'OPENALEX_MAILTO':<22} ✅ set ({openalex_mail})   → OpenAlex academic search")
else:
    print(f"  {'OPENALEX_MAILTO':<22} ℹ️  not set   → OpenAlex will use anonymous pool (lower rate limits)")
    print(f"  {'':22}    Set any email in .env: OPENALEX_MAILTO=your@email.com")

hunter_key = os.getenv("HUNTER_API_KEY", "")
if hunter_key:
    print(f"  {'HUNTER_API_KEY':<22} ✅ set   → Hunter.io email enrichment")
else:
    print(f"  {'HUNTER_API_KEY':<22} ℹ️  not set   → Hunter.io email enrichment disabled (optional)")

reddit_id = os.getenv("REDDIT_CLIENT_ID", "")
reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
if reddit_id and reddit_secret:
    print(f"  {'REDDIT_CLIENT_ID/SECRET':<22} ✅ set   → Reddit PRAW authenticated access")
else:
    print(f"  {'REDDIT_CLIENT_ID/SECRET':<22} ℹ️  not set   → Reddit scraper uses public JSON fallback (optional)")

# ── Package imports ───────────────────────────────────────────────
print("\n[DEPENDENCIES — Required Packages]")
packages = [
    ("flask",              "Flask web server",         True),
    ("requests",           "HTTP client",              True),
    ("bs4",                "BeautifulSoup HTML parse", True),
    ("lxml",               "Fast HTML parser",         True),
    ("cv2",                "OpenCV — image processing",True),
    ("numpy",              "NumPy — arrays",           True),
    ("dotenv",             "python-dotenv",            True),
    ("deepface",           "DeepFace — face matching", True),
    ("rapidfuzz",          "Fuzzy name matching",      True),
    ("google_lens_python", "Google Lens direct upload",False),
]
missing_critical = []
for pkg, desc, required in packages:
    try:
        __import__(pkg)
        print(f"  {pkg:<22} ✅ installed   → {desc}")
    except ImportError:
        mark = "❌ MISSING" if required else "⚠️  optional"
        print(f"  {pkg:<22} {mark}   → {desc}")
        if required:
            missing_critical.append(pkg)

# Sherlock — called as subprocess by username.py
import shutil as _shutil
sherlock_path = _shutil.which("sherlock")
if sherlock_path:
    print(f"  {'sherlock':<22} ✅ found at {sherlock_path}   → username scraper Layer 1")
else:
    print(f"  {'sherlock':<22} ⚠️  not in PATH   → username scraper Layer 1 will be skipped")
    print(f"  {'':22}    Install: pip install sherlock-project")

# ── Data / models directory ───────────────────────────────────────
print("\n[DATA]")
from pathlib import Path as _Path
models_dir = _Path("data/models")
if models_dir.exists() and any(models_dir.iterdir()):
    model_files = list(models_dir.iterdir())
    print(f"  OK    data/models/ exists with {len(model_files)} item(s) ✅")
else:
    print("  INFO  data/models/ empty or missing — DeepFace models will download on first run (~600MB)")

# ── SQLite WAL test ───────────────────────────────────────────────
print("\n[STORAGE]")
import sqlite3 as _sqlite3, tempfile as _tempfile, os as _os2
with _tempfile.NamedTemporaryFile(suffix='.db', delete=False) as _tf:
    _tmp_db = _tf.name
try:
    _conn = _sqlite3.connect(_tmp_db)
    _mode = _conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    _conn.close()
    if _mode == "wal":
        print("  OK    SQLite WAL mode ✅")
    else:
        print(f"  WARN  SQLite journal_mode={_mode} (expected WAL)")
finally:
    _os2.unlink(_tmp_db)

# ── Network connectivity ──────────────────────────────────────────
print("\n[Network Connectivity]")
import requests as req
tests = [
    ("https://catbox.moe",                "catbox.moe (image host, no key)"),
    ("https://html.duckduckgo.com/html/",  "DuckDuckGo HTML (LinkedIn search)"),
    ("https://www.bing.com/search",        "Bing HTML (fallback search)"),
    ("https://yandex.com/images/",         "Yandex (face reverse search)"),
    ("https://serpapi.com",                "SerpApi"),
]
for url, desc in tests:
    try:
        r = req.get(url, timeout=5,
                    headers={"User-Agent": "Mozilla/5.0"},
                    params={"q": "test"} if "search" in url else {})
        print(f"  {'✅' if r.ok else '⚠️ '} {desc} ({r.status_code})")
    except Exception as e:
        print(f"  ❌ {desc} — {e}")

# ── Image hosting test ────────────────────────────────────────────
print("\n[Image Hosting Test]")
# Create a tiny 1x1 white JPEG to test
import io
try:
    import cv2
    import numpy as np
    tiny = np.ones((10,10,3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", tiny)
    img_bytes = buf.tobytes()

    hosting_services = [
        ("imgbb.com",      lambda b: _test_imgbb(b)),
        ("catbox.moe",     lambda b: _test_catbox(b)),
        ("litterbox",      lambda b: _test_litterbox(b)),
        ("0x0.st",         lambda b: _test_0x0(b)),
    ]

    def _test_imgbb(b):
        key = os.getenv("IMGBB_API_KEY","")
        if not key: return None, "no IMGBB_API_KEY"
        import base64
        r = req.post("https://api.imgbb.com/1/upload",
                     data={"key": key, "image": base64.b64encode(b).decode(),
                           "expiration": 60}, timeout=10)
        url = r.json().get("data",{}).get("url","")
        return url or None, r.status_code

    def _test_catbox(b):
        r = req.post("https://catbox.moe/user/api.php",
                     data={"reqtype": "fileupload", "userhash": ""},
                     files={"fileToUpload": ("t.jpg", b, "image/jpeg")}, timeout=12)
        url = r.text.strip() if r.ok and r.text.startswith("https") else ""
        return url or None, r.status_code if r.ok else "failed"

    def _test_litterbox(b):
        r = req.post("https://litterbox.catbox.moe/resources/internals/api.php",
                     data={"reqtype":"fileupload","time":"1h"},
                     files={"fileToUpload": ("t.jpg", b, "image/jpeg")}, timeout=12)
        url = r.text.strip() if r.ok and r.text.startswith("https") else ""
        return url or None, r.status_code if r.ok else "failed"

    def _test_0x0(b):
        r = req.post("https://0x0.st",
                     files={"file": ("t.jpg", b, "image/jpeg")}, timeout=12)
        url = r.text.strip() if r.ok and r.text.startswith("http") else ""
        return url or None, r.status_code if r.ok else "failed"

    for name, fn in hosting_services:
        try:
            url, status = fn(img_bytes)
            if url:
                print(f"  ✅ {name} → {url[:50]}")
            else:
                print(f"  ❌ {name} → status {status}")
        except Exception as e:
            print(f"  ❌ {name} → {e}")
except Exception as e:
    print(f"  ⚠️  Could not run hosting tests: {e}")

# ── LinkedIn search test ──────────────────────────────────────────
print("\n[LinkedIn Search Test — searching 'Sundar Pichai site:linkedin.com/in']")
try:
    query = '"Sundar Pichai" site:linkedin.com/in'
    r = req.get("https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                       "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"},
                timeout=12)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for div in soup.select(".result__body, .result")[:5]:
        a = div.select_one(".result__title a, a.result__a")
        if a:
            href = a.get("href","")
            if "uddg=" in href:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = urllib.parse.unquote(qs.get("uddg",[""])[0])
            if "linkedin" in href:
                links.append(href)
    if links:
        print(f"  ✅ DDG LinkedIn search works: {len(links)} results")
        for l in links[:2]: print(f"     {l}")
    else:
        print("  ⚠️  DDG returned 0 LinkedIn results (may be rate-limited, try again)")
except Exception as e:
    print(f"  ❌ DDG LinkedIn test failed: {e}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SUMMARY")
print("═"*60)

cse_key = os.getenv("GOOGLE_CSE_KEY","")
cse_id  = os.getenv("GOOGLE_CSE_ID","")
serp    = os.getenv("SERPAPI_KEY","")
imgbb   = os.getenv("IMGBB_API_KEY","")

if missing_critical:
    pip_names = {"cv2":"opencv-python","bs4":"beautifulsoup4","dotenv":"python-dotenv"}
    install = " ".join(pip_names.get(p,p) for p in missing_critical)
    print(f"\n🔴 INSTALL MISSING PACKAGES FIRST:")
    print(f"   pip install {install}")
    if "google_lens_python" not in [p for p,_,_ in packages if True]:
        pass
    print(f"   pip install google-lens-python")

if not cse_key or not cse_id:
    print("\n⚡ GET GOOGLE CSE KEY (MOST IMPORTANT — enables LinkedIn search):")
    print("   1. Go to: https://programmablesearchengine.google.com")
    print("   2. Create engine → select 'Search the entire web'")
    print("   3. Go to: https://console.cloud.google.com/apis/credentials")
    print("   4. Create API key → enable 'Custom Search API'")
    print("   5. Add to .env: GOOGLE_CSE_KEY=... and GOOGLE_CSE_ID=...")

if not serp:
    print("\n⚡ GET SERPAPI KEY (enables Google Lens face search — free 100/mo):")
    print("   1. Go to: https://serpapi.com/users/sign_up")
    print("   2. Add to .env: SERPAPI_KEY=...")

if not imgbb:
    print("\n⚡ GET IMGBB KEY (needed for SerpApi image upload — free):")
    print("   1. Go to: https://imgbb.com → Account → API")
    print("   2. Add to .env: IMGBB_API_KEY=...")

if cse_key and cse_id:
    print("\n✅ Google CSE configured — LinkedIn search will work")
if serp and imgbb:
    print("✅ SerpApi + imgbb configured — face reverse search will work")
if not missing_critical:
    print("✅ All required packages installed")

print()