#!/usr/bin/env python3
"""Decide which STARK-PLACE packages to release, and (optionally) bump them.

Each package under packages/* is a standalone distribution with its own version
and its own `<dist-name>-v<version>` tags. This computes the release matrix:

  include a package when ANY of:
    * src changed          — files under its import dir changed since --base
    * manual bump          — its pyproject version has no matching tag yet
    * dispatch selection    — named in --dispatch (or --dispatch=all)

  then, BEFORE building, if the package's current version is already released
  (a tag equals it), auto-bump patch. A manually-bumped version (no tag yet) is
  used as-is — never double-bumped.

Emits a JSON matrix on stdout (and to $GITHUB_OUTPUT as `matrix`/`any` when set).
With --apply-bumps it also rewrites the bumped versions into the pyproject files
so a single commit can carry all bumps before the parallel build matrix runs.

Stdlib only. Runner Python (3.11+) has tomllib.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree
# What counts as "src" for auto-trigger: any changed file under the package dir
# EXCEPT these — docs/tests/changelog edits don't auto-trigger a release on their
# own (a pyproject change does: it means a version/dep change worth shipping).
SRC_EXCLUDE = ("tests/", "README", "CHANGELOG", "docs/")


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def dist_name(pyproject: Path) -> str | None:
    data = tomllib.loads(pyproject.read_text())
    return data.get("project", {}).get("name")


def cur_version(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text())
    return data["project"]["version"]


def latest_tag(name: str) -> tuple[str | None, str | None]:
    """Highest existing <name>-v* tag and its version (or None, None)."""
    tags = [t for t in sh("git", "tag", "-l", f"{name}-v*").splitlines() if t]
    if not tags:
        return None, None
    tags.sort(key=lambda t: _ver_key(t.rsplit("-v", 1)[-1]))
    top = tags[-1]
    return top, top.rsplit("-v", 1)[-1]


def _ver_key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v)[:3])


def src_changed(pkg_dir: Path, base: str) -> bool:
    diff = sh("git", "diff", "--name-only", f"{base}..HEAD", "--", str(pkg_dir))
    for line in diff.splitlines():
        if not line:
            continue
        rel = line[len(str(pkg_dir)) + 1:]
        if any(x in rel for x in SRC_EXCLUDE):
            continue
        return True  # a non-excluded file under the package changed → src change
    return False


def bump_patch(v: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", v)
    if not m:
        raise ValueError(f"cannot patch-bump non-semver version {v!r}")
    return f"{m[1]}.{m[2]}.{int(m[3]) + 1}{m[4]}"


def apply_version(pyproject: Path, new: str) -> None:
    text = pyproject.read_text()
    # Replace the first `version = "..."` (inside [project]).
    text2 = re.sub(r'(?m)^(version\s*=\s*")[^"]*(")', rf"\g<1>{new}\g<2>", text, count=1)
    pyproject.write_text(text2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages-dir", type=Path, default=Path("packages"))
    ap.add_argument("--base", default="", help="commit to diff against ('' = first run)")
    ap.add_argument("--dispatch", default="", help="'all', or comma-separated dist names")
    ap.add_argument("--apply-bumps", action="store_true")
    args = ap.parse_args()

    base = args.base or EMPTY_TREE
    dispatch = {s.strip() for s in args.dispatch.split(",") if s.strip()}
    dispatch_all = "all" in dispatch

    matrix = []
    for pyproject in sorted(args.packages_dir.glob("*/pyproject.toml")):
        name = dist_name(pyproject)
        if not name:
            continue  # examples / non-PEP621 → never released
        pkg_dir = pyproject.parent
        version = cur_version(pyproject)
        tag, tag_ver = latest_tag(name)

        selected = dispatch_all or name in dispatch
        changed = src_changed(pkg_dir, base)
        unreleased = tag_ver is None or version != tag_ver  # first release or manual bump

        if dispatch and not selected:
            continue
        if not (selected or changed or unreleased):
            continue

        if tag_ver is not None and version == tag_ver:
            new = bump_patch(version)  # already released → bump before build
            bumped = True
            if args.apply_bumps:
                apply_version(pyproject, new)
        else:
            new = version  # first release or explicit manual bump — as-is
            bumped = False

        matrix.append({
            "package": name,
            "dir": str(pkg_dir),
            "version": new,
            "tag": f"{name}-v{new}",
            "prev_tag": tag or "",
            "bumped": bumped,
        })

    out = {"include": matrix}
    print(json.dumps(out, indent=2))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"matrix={json.dumps(out)}\n")
            f.write(f"any={'true' if matrix else 'false'}\n")


if __name__ == "__main__":
    main()
