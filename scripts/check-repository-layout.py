#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Fail when tracked repository content escapes its documented ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = {
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "CLA.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "go.mod",
    "go.sum",
    "pyproject.toml",
    "requirements-dev.txt",
    "ruff.toml",
    "uv.lock",
    "versions.yaml",
}

ROOT_DIRECTORIES = {
    ".github",
    "benchmarks",
    "cmd",
    "deploy",
    "docs",
    "integrations",
    "internal",
    "native",
    "runtime",
    "scripts",
    "test",
    "third_party",
}

FORBIDDEN_PREFIXES = (
    "bin/",
    "build/",
    "dist/",
    "docs/old/",
    "docs/archive/",
    "native/nccl/experiments/",
    "internal/dockerapi/",
    "deploy/vllm/nccl/",
    "patches/",
    "profiles/",
    "runtime/python/",
    "sglang_plugin/",
    "vllm_disk_sleep/",
)


def repository_files() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    paths = [
        PurePosixPath(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    ]
    return [
        path
        for path in paths
        if os.path.lexists(REPOSITORY_ROOT / Path(path.as_posix()))
    ]


def violations(paths: list[PurePosixPath]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        rendered = path.as_posix()
        if path.parts[0] == "benchmarks" and not (
            rendered == "benchmarks/README.md"
            or (
                len(path.parts) == 3
                and path.parts[1] == "harnesses"
                and (path.name == "README.md" or path.suffix in {".py", ".sh"})
            )
        ):
            problems.append(f"benchmark history or output is repository content: {rendered}")
            continue
        if any(rendered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            problems.append(f"generated or retired path is repository content: {rendered}")
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            problems.append(f"Python bytecode is repository content: {rendered}")
            continue
        if any(part.endswith(".dist-info") for part in path.parts):
            problems.append(f"generated Python metadata is repository content: {rendered}")
            continue
        if len(path.parts) == 1:
            if rendered not in ROOT_FILES:
                problems.append(f"unowned root file: {rendered}")
            continue
        if path.parts[0] not in ROOT_DIRECTORIES:
            problems.append(f"unowned root directory: {path.parts[0]}/")
    return sorted(set(problems))


def main() -> int:
    problems = violations(repository_files())
    if problems:
        for problem in problems:
            print(f"repository-layout: {problem}", file=sys.stderr)
        return 1
    print("repository-layout: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
