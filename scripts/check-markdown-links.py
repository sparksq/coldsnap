#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Validate repository-local links in Markdown documentation."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("#", "http://", "https://", "mailto:")


def markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    paths = (
        REPOSITORY_ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )
    return sorted(path for path in paths if path.suffix == ".md" and path.is_file())


def link_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    return value.split(maxsplit=1)[0]


def missing_links(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for source in files:
        relative_source = source.relative_to(REPOSITORY_ROOT)
        for raw in MARKDOWN_LINK.findall(source.read_text(encoding="utf-8")):
            target = link_target(raw)
            if not target or target.startswith(EXTERNAL_PREFIXES):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            if target.startswith("/"):
                problems.append(f"{relative_source}: non-portable absolute link: {raw}")
                continue
            if not (source.parent / target).exists():
                problems.append(f"{relative_source}: missing local link: {raw}")
    return problems


def main() -> int:
    files = markdown_files()
    problems = missing_links(files)
    if problems:
        for problem in problems:
            print(f"markdown-links: {problem}", file=sys.stderr)
        return 1
    print(f"markdown-links: ok ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
