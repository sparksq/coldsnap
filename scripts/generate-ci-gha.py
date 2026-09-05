#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: BSD-3-Clause

"""Generate ColdSnap workflows from versions.yaml via scitrera-repo-tools."""

from __future__ import annotations

import os
import shutil
import sys
from typing import List

DEFAULT_SOURCE = "git+https://github.com/scitrera/repo-tools.git@b4aa219f66bfbc8c03e1c42644eb77ebb7983bd9"


def _try_uvx(args: List[str]) -> None:
    uv = shutil.which("uvx") or shutil.which("uv")
    if uv is None:
        return
    source = os.environ.get("REPO_TOOLS_SOURCE", DEFAULT_SOURCE)
    if os.path.basename(uv) == "uv":
        command = [uv, "tool", "run", "--from", source, "generate-ci-gha", *args]
    else:
        command = [uv, "--from", source, "generate-ci-gha", *args]
    os.execvp(command[0], command)


def _try_import(args: List[str]) -> bool:
    try:
        from scitrera_repo_tools.ci_gen_gha.cli import main as generate_main
    except ImportError:
        return False
    sys.argv = ["generate-ci-gha", *args]
    generate_main()
    return True


def main(arguments: List[str]) -> int:
    if _try_import(arguments):
        return 0
    _try_uvx(arguments)
    source = os.environ.get("REPO_TOOLS_SOURCE", DEFAULT_SOURCE)
    sys.stderr.write(
        "scitrera-repo-tools is unavailable. Install it or set REPO_TOOLS_SOURCE.\n"
        f"Suggested source: {source}\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
