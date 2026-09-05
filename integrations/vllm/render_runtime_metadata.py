#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Render minimal inference-engine plugin metadata from pyproject.toml."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_DISTRIBUTION = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_ENTRY_POINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$"
)


@dataclass(frozen=True)
class Project:
    name: str
    version: str
    description: str
    license_expression: str
    entry_point: str
    entry_point_group: str

    @property
    def dist_info(self) -> str:
        normalized = re.sub(r"[-_.]+", "_", self.name)
        return f"{normalized}-{self.version}.dist-info"


def _string_assignment(line: str) -> tuple[str, str] | None:
    name, separator, encoded = line.partition("=")
    if not separator:
        return None
    name = name.strip()
    encoded = encoded.strip()
    if not name or not encoded.startswith('"'):
        return None
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"unsupported TOML string assignment: {line!r}") from error
    if not isinstance(value, str):
        raise ValueError(f"TOML assignment is not a string: {line!r}")
    return name, value


def load_project(path: Path) -> Project:
    section = ""
    project: dict[str, str] = {}
    entry_points: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        assignment = _string_assignment(line)
        if assignment is None:
            continue
        name, value = assignment
        if section == "project" and name in {
            "name",
            "version",
            "description",
            "license",
        }:
            project[name] = value
        elif section in {
            'project.entry-points."vllm.general_plugins"',
            'project.entry-points."sglang.srt.plugins"',
        }:
            entry_points[name] = value
            project["entry_point_group"] = section.removeprefix(
                'project.entry-points."'
            ).removesuffix('"')

    result = Project(
        name=project.get("name", ""),
        version=project.get("version", ""),
        description=project.get("description", ""),
        license_expression=project.get("license", ""),
        entry_point=entry_points.get("coldsnap", ""),
        entry_point_group=project.get("entry_point_group", ""),
    )
    if not _DISTRIBUTION.fullmatch(result.name):
        raise ValueError("plugin project name is missing or invalid")
    if not _VERSION.fullmatch(result.version):
        raise ValueError("plugin project version must be a three-part release")
    if not result.description or "\n" in result.description:
        raise ValueError("plugin project description is missing or invalid")
    if result.license_expression != "AGPL-3.0-only":
        raise ValueError("ColdSnap plugin license must be AGPL-3.0-only")
    if not _ENTRY_POINT.fullmatch(result.entry_point):
        raise ValueError("ColdSnap plugin entry point is missing or invalid")
    if result.entry_point_group not in {
        "vllm.general_plugins",
        "sglang.srt.plugins",
    }:
        raise ValueError("ColdSnap plugin entry-point group is missing or invalid")
    return result


def render(project: Project) -> dict[str, bytes]:
    prefix = project.dist_info
    return {
        f"{prefix}/METADATA": (
            "Metadata-Version: 2.4\n"
            f"Name: {project.name}\n"
            f"Version: {project.version}\n"
            f"Summary: {project.description}\n"
            f"License-Expression: {project.license_expression}\n"
        ).encode(),
        f"{prefix}/entry_points.txt": (
            f"[{project.entry_point_group}]\n"
            f"coldsnap = {project.entry_point}\n"
        ).encode(),
    }


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(project_path: Path, output: Path) -> Path:
    project = load_project(project_path)
    for relative, data in render(project).items():
        _write_atomic(output / relative, data)
    return output / project.dist_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.project, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
