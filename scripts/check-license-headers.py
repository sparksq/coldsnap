#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Check or mechanically install the repository's first-party license headers."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT_LINES = (
    "SPDX-FileCopyrightText: 2026 Scitrera LLC",
    "SPDX-FileCopyrightText: 2026 Fox Engine Ltd",
)
DEFAULT_LICENSE = "AGPL-3.0-only"
LICENSE_OVERRIDES = {
    "scripts/generate-ci-gha.py": "BSD-3-Clause",
    "scripts/update-versions.py": "BSD-3-Clause",
}

# Machine-generated or third-party-derived content either cannot carry comments
# or must retain its upstream byte representation.
EXCLUDED_PREFIXES = (
    "native/nccl/releases/2.30.7-1/patches/",
    "native/nccl/releases/2.31.2-1/patches/",
    "third_party/licenses/",
)
EXCLUDED_PATHS = {
    ".github/workflows/publish-go.yml",
    ".github/workflows/version-check.yml",
    "CLA.md",
    "LICENSE",
    "go.sum",
    "cmd/coldsnap-criu-rpc/go.sum",
    "uv.lock",
}
HASH_BASENAMES = {
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "Makefile",
    "requirements-dev.txt",
}
HASH_SUFFIXES = {".sh", ".toml", ".yaml", ".yml"}
SLASH_SUFFIXES = {".c", ".cc", ".cpp", ".go", ".h", ".mod"}


def repository_files() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        PurePosixPath(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    )


def comment_style(path: PurePosixPath) -> str | None:
    rendered = path.as_posix()
    if rendered in EXCLUDED_PATHS or any(
        rendered.startswith(prefix) for prefix in EXCLUDED_PREFIXES
    ):
        return None
    if path.name.startswith("Dockerfile"):
        return "hash"
    if path.name in HASH_BASENAMES or path.suffix in HASH_SUFFIXES:
        return "hash"
    if path.suffix == ".py":
        return "hash"
    if path.suffix in SLASH_SUFFIXES:
        return "slash"
    if path.suffix == ".md":
        return "html"
    return None


def expected_header(style: str, license_expression: str) -> str:
    lines = (*COPYRIGHT_LINES, f"SPDX-License-Identifier: {license_expression}")
    if style == "hash":
        return "".join(f"# {line}\n" for line in lines)
    if style == "slash":
        return "".join(f"// {line}\n" for line in lines)
    if style == "html":
        return "<!--\n" + "\n".join(lines) + "\n-->\n"
    raise ValueError(f"unsupported comment style: {style}")


def insertion_offset(path: PurePosixPath, text: str) -> int:
    offset = 0
    if text.startswith("#!"):
        newline = text.find("\n")
        offset = len(text) if newline < 0 else newline + 1
    if path.name.startswith("Dockerfile"):
        parser = re.match(r"#\s*syntax=[^\n]+\n", text)
        if parser:
            offset = parser.end()
    return offset


def remove_existing_spdx_header(text: str, offset: int, style: str) -> str:
    prefix, remainder = text[:offset], text[offset:]
    if style == "html":
        pattern = re.compile(
            r"<!--\n(?:SPDX-FileCopyrightText:[^\n]+\n)*"
            r"SPDX-License-Identifier:[^\n]+\n-->\n?"
        )
    else:
        # Accept either source-comment spelling so --fix can repair a file
        # whose existing header used the wrong syntax for its format.
        marker = r"(?:\#|//)"
        pattern = re.compile(
            rf"(?:{marker} SPDX-FileCopyrightText:[^\n]+\n)*"
            rf"{marker} SPDX-License-Identifier:[^\n]+\n?"
        )
    return prefix + pattern.sub("", remainder, count=1)


def licensed_text(path: PurePosixPath, text: str, style: str) -> str:
    offset = insertion_offset(path, text)
    text = remove_existing_spdx_header(text, offset, style)
    offset = insertion_offset(path, text)
    before, after = text[:offset], text[offset:]
    license_expression = LICENSE_OVERRIDES.get(path.as_posix(), DEFAULT_LICENSE)
    header = expected_header(style, license_expression)
    if style == "html":
        separator = "\n" if after and not after.startswith("\n") else ""
    else:
        separator = "\n" if after and not after.startswith("\n") else ""
    return before + header + separator + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="install or replace eligible file headers"
    )
    args = parser.parse_args()

    problems: list[str] = []
    changed: list[str] = []
    for relative in repository_files():
        style = comment_style(relative)
        if style is None:
            continue
        path = REPOSITORY_ROOT / Path(relative.as_posix())
        if not os.path.isfile(path):
            continue
        original = path.read_text(encoding="utf-8")
        expected = licensed_text(relative, original, style)
        if original == expected:
            continue
        if args.fix:
            path.write_text(expected, encoding="utf-8")
            changed.append(relative.as_posix())
        else:
            problems.append(relative.as_posix())

    if problems:
        for path in problems:
            print(f"license-header: missing or invalid: {path}", file=sys.stderr)
        return 1
    if args.fix:
        print(f"license-header: updated {len(changed)} file(s)")
    else:
        print("license-header: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
