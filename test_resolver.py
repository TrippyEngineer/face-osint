"""TDD for resolver identity-honesty (F1+F2). Run with Windows python from root.
Encodes the reverse-image-search + name-as-corroboration model."""
import aggregator.resolver as R


def m(**kw):
    d = {"source": "reverse_face", "combined_score": 0.5}
    d.update(kw)
    return d


# A reverse_face look-alike: name lives ONLY in `title` (not name/username),
# strong (self-referential) face score, a stranger's LinkedIn.
PUJYA = m(
    title="PUJYA GHOSH - Programme officer at Centre for World Solidarity | LinkedIn",
    url="https://mn.linkedin.com/in/pujya-ghosh",
    face_score=0.94, face_verified=True, verdict="confirmed",
    combined_score=0.83, photo_url="https://encrypted-tbn0.gstatic.com/x",
    email="pujya@example.org", company="Centre for World Solidarity",
)

fails = []
def check(label, cond):
    print(("  ok " if cond else "FAIL ") + label)
    if not cond: fails.append(label)


# ── A: named query, face hit whose name DISAGREES (the "PUJYA GHOSH" bug) ──
a = R.resolve("Neeraj Jain", [PUJYA])
print("A named+mismatch ->", {k: a.get(k) for k in
      ("verdict", "resolved_name", "corroborated", "profile_urls", "email")})
check("A.verdict not confirmed", a["verdict"] != "confirmed")
check("A.not corroborated", a.get("corroborated") is False)
check("A.no stranger url asserted", a.get("profile_urls") == [])
check("A.no stranger contact leaked", not a.get("email"))
check("A.resolved_name not the stranger", a.get("resolved_name") != "PUJYA GHOSH")
check("A.face_leads surfaced", bool(a.get("face_leads")) and
      a["face_leads"][0].get("name_on_page") == "PUJYA GHOSH")

# ── B: named query, face hit whose name AGREES (corroborated -> confirmed) ──
good = m(title="Neeraj Jain - Software Engineer | GitHub",
         url="https://github.com/neerajjain", face_score=0.92,
         face_verified=True, verdict="confirmed", combined_score=0.85)
b = R.resolve("Neeraj Jain", [good])
print("B named+match    ->", {k: b.get(k) for k in ("verdict", "resolved_name", "corroborated")})
check("B.corroborated", b.get("corroborated") is True)
check("B.verdict stays confirmed", b["verdict"] == "confirmed")
check("B.resolved_name correct", b.get("resolved_name") == "Neeraj Jain")
check("B.url present", "https://github.com/neerajjain" in b.get("profile_urls", []))

# ── C: photo-only (no query name) -> no identity claim, leads only ──
c = R.resolve("", [PUJYA])
print("C photo-only     ->", {k: c.get(k) for k in
      ("verdict", "resolved_name", "corroborated", "profile_urls")})
check("C.not confirmed", c["verdict"] != "confirmed")
check("C.no stranger url", c.get("profile_urls") == [])
check("C.no name asserted", c.get("resolved_name") in ("", "Unknown", None))
check("C.face_leads surfaced", bool(c.get("face_leads")))

# ── D: name-only cluster (no face) still works as a name lead ──
gh = m(source="github", name="Neeraj Jain", url="https://github.com/njain",
       verdict="possible", combined_score=0.30)
d = R.resolve("Neeraj Jain", [gh])
print("D name-only      ->", {k: d.get(k) for k in ("verdict", "resolved_name", "profile_urls")})
check("D.resolved_name kept", d.get("resolved_name") == "Neeraj Jain")
check("D.url kept", "https://github.com/njain" in d.get("profile_urls", []))
check("D.not confirmed", d["verdict"] != "confirmed")

print("\n" + ("ALL RESOLVER TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
