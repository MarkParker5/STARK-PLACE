#!/usr/bin/env python3
"""Generate the STARK-PLACE registry page — the pip `--find-links` target.

The registry is the intersection of:
  1. packages/*/pyproject.toml  -> canonical package list (name, description,
     requires-python, stark-place-core compat). Examples/WIP have no pyproject,
     so they never appear. This is the display-control lever.
  2. wheels                      -> version history, from either a local --dist
     dir (for testing) or a --assets JSON of GitHub release assets (in CI).

Output is static HTML: every wheel is a real <a href>, so pip scrapes all
versions (even those inside collapsed <details>) and resolves ranges natively.
pip runs no JS — nothing here depends on client-side scripting.

Stdlib only. Python 3.10+ (tomllib on 3.11+, falls back to tomli).
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib  # type: ignore


# --- wheel filename parsing -------------------------------------------------
# {distribution}-{version}(-{build})?-{pytag}-{abitag}-{platform}.whl  (PEP 427)
_WHEEL_RE = re.compile(
    r"^(?P<name>.+?)-(?P<version>[^-]+?)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<pytag>[^-]+)-(?P<abitag>[^-]+)-(?P<plat>[^-]+)\.whl$"
)


def _norm(name: str) -> str:
    """Normalize a distribution name for matching (PEP 503 + wheel underscores)."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Wheel:
    filename: str
    url: str
    dist: str          # normalized distribution name
    version: str
    pytag: str
    abitag: str
    plat: str

    @classmethod
    def parse(cls, filename: str, url: str) -> "Wheel | None":
        m = _WHEEL_RE.match(filename)
        if not m:
            return None
        return cls(
            filename=filename,
            url=url,
            dist=_norm(m["name"]),
            version=m["version"],
            pytag=m["pytag"],
            abitag=m["abitag"],
            plat=m["plat"],
        )


@dataclass
class Package:
    dist: str                      # normalized distribution name, e.g. stark-place-core
    display_name: str              # as declared in pyproject
    description: str
    requires_python: str
    core_compat: str               # declared stark-place-core spec, or ""
    wheels: list[Wheel] = field(default_factory=list)


# --- pyproject scan ---------------------------------------------------------
def scan_packages(packages_dir: Path) -> dict[str, Package]:
    packages: dict[str, Package] = {}
    for pyproject in sorted(packages_dir.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        proj = data.get("project")
        if not proj or "name" not in proj:
            continue  # not a PEP 621 distribution — skip
        name = proj["name"]
        core_compat = ""
        for dep in proj.get("dependencies", []):
            if _norm(dep).startswith("stark-place-core"):
                core_compat = dep.strip()
                break
        packages[_norm(name)] = Package(
            dist=_norm(name),
            display_name=name,
            description=proj.get("description", ""),
            requires_python=proj.get("requires-python", ""),
            core_compat=core_compat,
        )
    return packages


# --- wheel collection -------------------------------------------------------
def collect_wheels(dist_dir: Path | None, assets_file: Path | None) -> list[Wheel]:
    wheels: list[Wheel] = []
    if dist_dir:
        for whl in sorted(dist_dir.glob("*.whl")):
            w = Wheel.parse(whl.name, url=whl.name)  # relative href for local serving
            if w:
                wheels.append(w)
    if assets_file:
        assets = json.loads(assets_file.read_text())
        for a in assets:
            fn = a.get("name") or a.get("filename", "")
            url = a.get("browser_download_url") or a.get("url", "")
            if fn.endswith(".whl"):
                w = Wheel.parse(fn, url=url)
                if w:
                    wheels.append(w)
    return wheels


def _version_key(v: str) -> tuple:
    # Best-effort PEP 440-ish sort: numeric segments desc, pre-releases after.
    parts = re.split(r"[.\-+]", v)
    key = []
    for p in parts:
        m = re.match(r"^(\d+)(.*)$", p)
        if m:
            key.append((0, int(m.group(1)), m.group(2)))
        else:
            key.append((1, 0, p))
    return tuple(key)


# --- rendering --------------------------------------------------------------
_CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --muted:#666; --bg:#fff;
  --card:#f6f6f7; --border:#e2e2e4; --accent:#ef6c00; }
@media (prefers-color-scheme: dark) { :root { --fg:#e8e8ea; --muted:#9a9aa2;
  --bg:#0f0f11; --card:#18181b; --border:#2a2a2e; --accent:#ffa726; } }
* { box-sizing:border-box; } body { margin:0; padding:2rem 1rem; background:var(--bg);
  color:var(--fg); font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width:820px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; } .lede { color:var(--muted); margin:0 0 1.5rem; }
.pkg { background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:1rem 1.25rem; margin:0 0 1rem; }
.pkg h2 { font-size:1.15rem; margin:0 0 .25rem; } .pkg h2 code { font-size:1rem; }
.desc { color:var(--muted); margin:.25rem 0 .75rem; }
.meta { font-size:.8rem; color:var(--muted); margin:.25rem 0 .75rem; }
.meta span { display:inline-block; margin-right:1rem; }
.latest a, details a { color:var(--accent); text-decoration:none; font-family:ui-monospace,monospace;
  font-size:.85rem; word-break:break-all; }
.latest a:hover, details a:hover { text-decoration:underline; }
.row { display:flex; gap:.5rem; align-items:baseline; padding:.15rem 0; flex-wrap:wrap; }
.tag { font-size:.7rem; color:var(--muted); border:1px solid var(--border);
  border-radius:5px; padding:0 .4rem; }
details { margin-top:.5rem; } summary { cursor:pointer; color:var(--muted); font-size:.85rem; }
.vergroup { margin:.5rem 0 .25rem; padding-left:.5rem; border-left:2px solid var(--border); }
.vergroup > .vlabel { font-size:.8rem; font-weight:600; color:var(--fg); margin:.35rem 0 .1rem; }
#search { width:100%; padding:.6rem .8rem; margin:0 0 1.25rem; border:1px solid var(--border);
  border-radius:9px; background:var(--card); color:var(--fg); font-size:.95rem; }
#search:focus { outline:2px solid var(--accent); outline-offset:-1px; }
.pkg[hidden] { display:none; }
.noresult { color:var(--muted); }
code { background:rgba(128,128,128,.15); padding:.05rem .3rem; border-radius:4px; }
.cmd { display:block; background:rgba(128,128,128,.12); border:1px solid var(--border);
  border-radius:8px; padding:.5rem .75rem; margin:.5rem 0 0; font-family:ui-monospace,monospace;
  font-size:.8rem; overflow-x:auto; white-space:pre; }
footer { color:var(--muted); font-size:.8rem; margin-top:2rem; }
"""


def _wheel_row(w: Wheel) -> str:
    tags = f'<span class="tag">{html.escape(w.pytag)}</span>'
    if w.plat != "any":
        tags += f' <span class="tag">{html.escape(w.plat)}</span>'
    return (
        f'<div class="row"><a href="{html.escape(w.url)}">{html.escape(w.filename)}</a>{tags}</div>'
    )


def render(packages: list[Package], find_links_url: str, site_title: str) -> str:
    blocks = []
    for pkg in packages:
        if not pkg.wheels:
            continue
        wheels = sorted(pkg.wheels, key=lambda w: _version_key(w.version), reverse=True)
        latest_ver = wheels[0].version
        latest = [w for w in wheels if w.version == latest_ver]
        older = [w for w in wheels if w.version != latest_ver]

        meta = []
        if pkg.requires_python:
            meta.append(f'<span>Python <code>{html.escape(pkg.requires_python)}</code></span>')
        if pkg.core_compat:
            meta.append(f'<span>needs <code>{html.escape(pkg.core_compat)}</code></span>')
        meta_html = f'<div class="meta">{"".join(meta)}</div>' if meta else ""

        latest_html = '<div class="latest">' + "".join(_wheel_row(w) for w in latest) + "</div>"
        older_html = ""
        if older:
            older_versions = sorted(
                {w.version for w in older}, key=_version_key, reverse=True
            )
            groups = []
            for ver in older_versions:
                rows = "".join(_wheel_row(w) for w in older if w.version == ver)
                groups.append(
                    f'<div class="vergroup"><div class="vlabel">v{html.escape(ver)}</div>{rows}</div>'
                )
            older_html = (
                f"<details><summary>{len(older_versions)} older version(s)</summary>"
                f'{"".join(groups)}</details>'
            )
        install = (
            f"pip install {html.escape(pkg.display_name)} \\\n"
            f"  --find-links {html.escape(find_links_url)}"
        )
        search_key = html.escape(f"{pkg.display_name} {pkg.description}".lower())
        blocks.append(
            f'<div class="pkg" data-search="{search_key}">'
            f'<h2><code>{html.escape(pkg.display_name)}</code> '
            f'<small>v{html.escape(latest_ver)}</small></h2>'
            f'<p class="desc">{html.escape(pkg.description)}</p>'
            f"{meta_html}{latest_html}{older_html}"
            f'<code class="cmd">{install}</code></div>'
        )

    body = "\n".join(blocks) or "<p>No packages published yet.</p>"
    return (
        "<!DOCTYPE html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(site_title)}</title><style>{_CSS}</style></head><body>"
        f'<div class="wrap"><h1>{html.escape(site_title)}</h1>'
        '<p class="lede">Separately-installable S.T.A.R.K. Platform modules. '
        "Add the <code>--find-links</code> URL to any install.</p>"
        '<input id="search" type="search" placeholder="Filter packages…" '
        'autocomplete="off" aria-label="Filter packages">'
        f'<div id="list">{body}</div>'
        '<p class="noresult" id="noresult" hidden>No packages match.</p>'
        "<footer>This page is the pip <code>--find-links</code> target — every "
        "version above (including collapsed ones) is resolvable via native pip "
        "version ranges. No exact URLs needed.</footer></div>"
        "<script>(function(){var q=document.getElementById('search'),"
        "n=document.getElementById('noresult'),"
        "cards=[].slice.call(document.querySelectorAll('.pkg'));"
        "q.addEventListener('input',function(){var t=q.value.trim().toLowerCase(),shown=0;"
        "cards.forEach(function(c){var m=c.getAttribute('data-search').indexOf(t)>-1;"
        "c.hidden=!m;if(m)shown++;});n.hidden=shown>0;});})();</script>"
        "</body></html>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packages-dir", type=Path, default=Path("packages"))
    ap.add_argument("--dist", type=Path, help="local dir of .whl files (testing)")
    ap.add_argument("--assets", type=Path, help="JSON of release assets (CI)")
    ap.add_argument("--out", type=Path, default=Path("find-links/index.html"))
    ap.add_argument("--find-links-url", default="https://place.markparker.me/")
    ap.add_argument("--title", default="STARK-PLACE registry")
    args = ap.parse_args()

    packages = scan_packages(args.packages_dir)
    wheels = collect_wheels(args.dist, args.assets)

    for w in wheels:
        if w.dist in packages:  # list control: only wheels for a current package
            packages[w.dist].wheels.append(w)

    ordered = sorted(packages.values(), key=lambda p: p.dist)
    out_html = render(ordered, args.find_links_url, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_html)

    shown = [p.display_name for p in ordered if p.wheels]
    print(f"Wrote {args.out} — {len(shown)} package(s): {', '.join(shown) or '(none)'}")
    dropped = {w.dist for w in wheels} - set(packages)
    if dropped:
        print(f"Ignored wheels for non-listed packages: {', '.join(sorted(dropped))}")


if __name__ == "__main__":
    main()
