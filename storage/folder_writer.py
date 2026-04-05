"""
storage/folder_writer.py
─────────────────────────
Creates and manages the person output folder.

Output structure:
    data/output/
    └── John_Doe_20260303_1432_a1b2c3/
        ├── captured_photo.jpg     ← original camera frame
        ├── face_crop.jpg          ← 160x160 aligned face crop
        ├── info.txt               ← human-readable report (PRIMARY OUTPUT)
        ├── matches_summary.json   ← full structured data
        └── scraped_photos/        ← downloaded profile photos
            ├── github_johndoe.jpg
            └── ...

info.txt is designed to be read at a glance, shared by email, or printed.
No JSON, no parsing — a clean plaintext report.
"""

import os
import re
import json
import hashlib
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

import config

logger = logging.getLogger(__name__)


class FolderWriter:
    def __init__(self, output_dir: Path = config.OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Create folder immediately on capture ─────────────────────────────
    def create_folder(
        self,
        name:      str,
        search_id: str,
        frame:     np.ndarray,
        face_crop: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Called the moment the user captures a face.
        Writes captured_photo.jpg + face_crop.jpg + placeholder info.txt
        so the folder appears immediately in the filesystem.
        """
        folder = self.output_dir / self._folder_name(name, search_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scraped_photos").mkdir(exist_ok=True)

        cv2.imwrite(str(folder / "captured_photo.jpg"), frame)

        if face_crop is not None:
            cv2.imwrite(str(folder / "face_crop.jpg"), face_crop)

        self._write_placeholder(folder, name, search_id)

        logger.info(f"Folder created: {folder}")
        return folder

    # ── Write final results after all scrapers complete ──────────────────
    def write_results(
        self,
        folder:      Path,
        name:        str,
        search_id:   str,
        all_results: dict,
        identity:    dict,
    ):
        """
        Overwrites placeholder info.txt with full report.
        Downloads scraped profile photos.
        Writes matches_summary.json.
        """
        self._save_scraped_photos(folder, all_results)
        self._write_info_txt(folder, name, search_id, all_results, identity)
        self._write_json(folder, name, search_id, all_results, identity)
        logger.info(f"Results written → {folder}")

    # ── List / read ───────────────────────────────────────────────────────
    def list_all(self) -> list:
        result = []
        for entry in sorted(self.output_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            jf = entry / "matches_summary.json"
            if jf.exists():
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    result.append({
                        "folder":    entry.name,
                        "search_id": data.get("search_id"),
                        "name":      data.get("name"),
                        "verdict":   data.get("identity", {}).get("verdict"),
                        "path":      str(entry),
                    })
                except Exception:
                    pass
        return result

    # ── Internal helpers ──────────────────────────────────────────────────
    def _folder_name(self, name: str, search_id: str) -> str:
        # Strip characters that are invalid in Windows filenames
        safe = re.sub(r'[\\/:*?"<>|]', "_", name)
        safe = safe.replace(" ", "_").strip("._")[:40]
        if not safe:
            safe = "unknown"
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        sid = search_id[:6]
        return f"{safe}_{ts}_{sid}"

    def _write_placeholder(self, folder: Path, name: str, search_id: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"NAME:        {name}\n"
            f"SEARCH ID:   {search_id}\n"
            f"DATE:        {ts}\n"
            f"STATUS:      Processing — scraping in progress...\n\n"
            f"This file will be updated when all scrapers complete.\n"
        )
        (folder / "info.txt").write_text(text, encoding="utf-8")

    def _write_info_txt(
        self,
        folder:      Path,
        name:        str,
        search_id:   str,
        all_results: dict,
        identity:    dict,
    ):
        S   = "═" * 58
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        verdict  = identity.get("verdict", "unknown").upper()
        score    = identity.get("combined_score", 0)
        face_sc  = identity.get("face_score")
        sources  = identity.get("sources", [])

        lines += [
            S,
            "  FACE OSINT REPORT",
            S,
            f"  NAME:            {name}",
            f"  SEARCH ID:       {search_id}",
            f"  DATE:            {ts}",
            f"  VERDICT:         {verdict}",
            f"  COMBINED SCORE:  {score:.3f}",
            f"  FACE SCORE:      {face_sc:.3f}" if face_sc else "  FACE SCORE:      N/A",
            f"  SOURCES HIT:     {', '.join(sources) if sources else 'none'}",
        ]

        # Resolved identity summary
        if identity.get("resolved_name") and identity["resolved_name"] != name:
            lines.append(f"  RESOLVED AS:     {identity['resolved_name']}")
        if identity.get("email"):
            lines.append(f"  EMAIL:           {identity['email']}")
        if identity.get("company"):
            lines.append(f"  EMPLOYER:        {identity['company']}")
        if identity.get("location"):
            lines.append(f"  LOCATION:        {identity['location']}")
        if identity.get("profile_urls"):
            lines.append(f"  PROFILES:")
            for url in identity["profile_urls"][:8]:
                lines.append(f"    • {url}")
        lines.append("")

        # ── Per-source sections ───────────────────────────────────────────
        section_map = {
            "search_engines": ("SEARCH ENGINES",      _fmt_search),
            "reverse_face":   ("REVERSE FACE SEARCH", _fmt_reverse_image),
            "academic":       ("ACADEMIC",             _fmt_academic),
            "github":         ("GITHUB",               _fmt_github),
            "reddit":         ("REDDIT",               _fmt_reddit),
            "passive":        ("PASSIVE INTEL",        _fmt_passive),
            "username":       ("USERNAME SEARCH",      _fmt_username),
        }

        for key, (title, formatter) in section_map.items():
            data = all_results.get(key, {})
            if not data or data.get("error"):
                if data.get("error"):
                    lines.append(f"─── {title} {'─' * (54 - len(title))}")
                    lines.append(f"  Error: {data['error'][:120]}")
                    lines.append("")
                continue
            section_lines = formatter(data)
            if section_lines:
                lines.append(f"─── {title} {'─' * (54 - len(title))}")
                lines.extend(section_lines)
                lines.append("")

        lines += [S, "  END OF REPORT", S]
        (folder / "info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_json(self, folder, name, search_id, all_results, identity):
        payload = {
            "search_id":  search_id,
            "name":       name,
            "identity":   identity,
            "results":    {
                k: v for k, v in all_results.items()
                if not k.startswith("_")
            },
        }
        (folder / "matches_summary.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def _save_scraped_photos(self, folder: Path, all_results: dict):
        photos_dir = folder / "scraped_photos"
        photos_dir.mkdir(exist_ok=True)
        seen_urls: set = set()   # skip duplicate photo URLs across all sources

        for source, data in all_results.items():
            if not isinstance(data, dict):
                continue
            for match in data.get("matches", []):
                url = match.get("photo_url") or match.get("avatar_url") or match.get("preview_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    r = requests.get(
                        url,
                        headers = config.BROWSER_HEADERS,
                        timeout = 8,
                        stream  = True,
                    )
                    r.raise_for_status()
                    if "image" not in r.headers.get("content-type", ""):
                        continue
                    uname    = re.sub(r'[\\/:*?"<>|]', "_", match.get("username") or "photo")[:30]
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filepath = photos_dir / f"{source}_{uname}_{url_hash}.jpg"
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                except Exception as e:
                    logger.debug(f"Could not download photo from {source}: {e}")


# ── Per-source formatters ─────────────────────────────────────────────────
def _fmt_search(data: dict) -> list:
    lines = []
    for m in data.get("matches", [])[:8]:
        conf = m.get("source_confidence")
        conf_str = f"  conf={conf*100:.0f}%" if conf is not None else ""
        lines.append(f"  • {m.get('url', '')}{conf_str}")
        if m.get("snippet"):
            lines.append(f"    {m['snippet'][:100]}")
        if m.get("linkedin_headline"):
            lines.append(f"    Headline: {m['linkedin_headline'][:120]}")
        if m.get("linkedin_location"):
            lines.append(f"    Location: {m['linkedin_location'][:80]}")
    return lines


def _fmt_reverse_image(data: dict) -> list:
    matches = data.get("matches", [])
    if not matches:
        return ["  No image matches found."]
    lines = [f"  Found {len(matches)} result(s):"]
    for m in matches[:6]:
        score = f"  face={m['face_score']:.3f}" if m.get("face_score") else ""
        lines.append(f"  [{m.get('source','')}]{score}  {m.get('url','')}")
    return lines


def _fmt_academic(data: dict) -> list:
    lines = []
    for m in data.get("matches", [])[:3]:
        lines.append(f"  Source:       {m.get('source', '')}")
        if m.get("name"):        lines.append(f"  Name:         {m['name']}")
        if m.get("affiliation"): lines.append(f"  Affiliation:  {m['affiliation']}")
        if m.get("paper_count"): lines.append(f"  Papers:       {m['paper_count']}")
        if m.get("h_index"):     lines.append(f"  h-index:      {m['h_index']}")
        if m.get("profile_url"): lines.append(f"  Profile:      {m['profile_url']}")
        lines.append("")
    return lines


def _fmt_github(data: dict) -> list:
    lines = []
    for m in data.get("matches", [])[:3]:
        if m.get("username"):  lines.append(f"  Username:     {m['username']}")
        if m.get("name"):      lines.append(f"  Name:         {m['name']}")
        if m.get("company"):   lines.append(f"  Company:      {m['company']}")
        if m.get("location"):  lines.append(f"  Location:     {m['location']}")
        if m.get("email"):     lines.append(f"  Email:        {m['email']}")
        if m.get("repos") is not None: lines.append(f"  Repos:        {m['repos']}")
        if m.get("bio"):       lines.append(f"  Bio:          {m['bio'][:100]}")
        lines.append(f"  URL:          github.com/{m.get('username','')}")
        lines.append("")
    return lines


def _fmt_reddit(data: dict) -> list:
    lines = []
    for m in data.get("matches", [])[:3]:
        if m.get("username"): lines.append(f"  Username:     u/{m['username']}")
        if m.get("karma"):    lines.append(f"  Karma:        {m['karma']:,}")
        if m.get("top_subreddits"):
            lines.append(f"  Active in:    {', '.join(m['top_subreddits'][:6])}")
        lines.append("")
    return lines


def _fmt_passive(data: dict) -> list:
    lines = []
    if data.get("wayback_urls"):
        lines.append(f"  Wayback Machine: {len(data['wayback_urls'])} archived URLs")
        for u in data["wayback_urls"][:3]:
            lines.append(f"    • {u}")
    if data.get("gdelt_mentions"):
        lines.append(f"  News mentions (GDELT): {data['gdelt_mentions']}")
    if data.get("crt_domains"):
        lines.append(f"  Certificate domains:   {', '.join(data['crt_domains'][:5])}")
    if data.get("matches"):
        hunter_hits = [m for m in data["matches"] if m.get("source") == "hunter"]
        if hunter_hits:
            lines.append(f"  Hunter.io email: {hunter_hits[0].get('email', '')}")
        # Email expansion results
        for m in data["matches"]:
            if m.get("gravatar_url"):
                lines.append(f"  Gravatar avatar:  {m['gravatar_url']}")
            if m.get("gravatar_profile"):
                lines.append(f"  Gravatar profile: {m['gravatar_profile']}")
            if m.get("email_domain"):
                lines.append(f"  Email domain:     {m['email_domain']}")
    return lines


def _fmt_username(data: dict) -> list:
    lines = []
    matches = data.get("matches", [])[:8]
    if not matches:
        return ["  No username matches found."]
    for m in matches:
        uname = m.get("username", "")
        url   = m.get("url", "")
        src   = m.get("source", "")
        if uname:
            lines.append(f"  \u2022 {uname:<20} {url}  [{src}]")
    return lines
