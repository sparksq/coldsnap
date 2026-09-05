#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Plan immutable, multi-architecture NCCL payload publications."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASES_ROOT = ROOT / "native/nccl/releases"
PROVIDER_ID = re.compile(
    r"^nccl-(?P<release>[0-9]+\.[0-9]+\.[0-9]+-[0-9]+)\+coldsnap\.(?P<revision>[0-9]+)$"
)
RUNNERS = {
    "linux/amd64": "ubuntu-24.04",
    "linux/arm64": "ubuntu-24.04-arm",
}
GLOBAL_BUILD_INPUTS = {
    ".dockerignore",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "deploy/nccl/Dockerfile.payload",
    "scripts/nccl-nvcc-reproducible.sh",
    "third_party/licenses/nccl/LICENSE.txt",
}
GLOBAL_BUILD_PREFIXES = (
    "native/nccl/abi/",
    "native/nccl_checkpoint_coord/",
)
NON_PAYLOAD_RELEASE_FILES = {
    "qualification.json",
    "source-diff.md",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _release_metadata(release_root: Path) -> dict[str, Any]:
    recipe = _load_json(release_root / "recipe.json")
    source_lock = _load_json(release_root / "source.lock")
    release = release_root.name
    match = PROVIDER_ID.fullmatch(str(recipe.get("provider_id", "")))
    if match is None or match.group("release") != release:
        raise ValueError(f"{release_root}: provider_id does not match the release directory")
    if recipe.get("format") != 2 or recipe.get("kind") != "coldsnap-nccl-provider-recipe":
        raise ValueError(f"{release_root}: unsupported provider recipe")
    builder = recipe.get("builder")
    build = recipe.get("build")
    if not isinstance(builder, dict) or not isinstance(build, dict):
        raise ValueError(f"{release_root}: incomplete provider build metadata")
    platforms = builder.get("payload_platforms")
    if platforms != sorted(RUNNERS):
        raise ValueError(
            f"{release_root}: payload_platforms must be {sorted(RUNNERS)!r}"
        )
    build_image = str(builder.get("payload_build_image", ""))
    if re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", build_image) is None:
        raise ValueError(f"{release_root}: payload_build_image must be digest-pinned")
    repository = str(builder.get("payload_repository", ""))
    if (
        re.fullmatch(
            r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/[a-z0-9._-]+(?:/[a-z0-9._-]+)*",
            repository,
        )
        is None
    ):
        raise ValueError(f"{release_root}: payload_repository is invalid")
    provider_tag = f"{release}.coldsnap.{match.group('revision')}"
    common = {
        "release": release,
        "release_key": re.sub(r"[^A-Za-z0-9]+", "-", release).strip("-"),
        "provider_id": recipe["provider_id"],
        "provider_tag": provider_tag,
        "release_alias": release,
        "repository": repository,
        "payload_dockerfile": builder["payload_dockerfile"],
        "payload_build_image": build_image,
        "nccl_library_release": str(recipe["nccl_release"]),
        "nccl_version_code": str(recipe["nccl_version_code"]),
        "nvcc_gencode": str(build["nvcc_gencode"]),
        "reproducible_nvcc": "1" if build["use_reproducible_nvcc"] else "0",
        "strip_outputs": "1" if build["strip_unneeded"] else "0",
        "source_archive": str(source_lock["source_archive"]),
        "source_archive_sha256": str(source_lock["source_archive_sha256"]),
    }
    if not common["nvcc_gencode"]:
        raise ValueError(f"{release_root}: nvcc_gencode is empty")
    return common


def _all_releases() -> dict[str, dict[str, Any]]:
    releases: dict[str, dict[str, Any]] = {}
    for recipe_path in sorted(RELEASES_ROOT.glob("*/recipe.json")):
        metadata = _release_metadata(recipe_path.parent)
        releases[metadata["release"]] = metadata
    if not releases:
        raise ValueError("no NCCL provider releases found")
    return releases


def _changed_paths(base_ref: str, head_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}..{head_ref}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _select_changed(
    releases: dict[str, dict[str, Any]], changed_paths: Iterable[str]
) -> set[str]:
    changed = set(changed_paths)
    if any(path in GLOBAL_BUILD_INPUTS for path in changed) or any(
        path.startswith(prefix) for path in changed for prefix in GLOBAL_BUILD_PREFIXES
    ):
        return set(releases)
    selected: set[str] = set()
    release_prefix = "native/nccl/releases/"
    for path in changed:
        if not path.startswith(release_prefix):
            continue
        parts = path[len(release_prefix) :].split("/", 1)
        if len(parts) != 2 or parts[0] not in releases:
            continue
        if parts[1] not in NON_PAYLOAD_RELEASE_FILES:
            selected.add(parts[0])
    return selected


def _make_plan(selected: Iterable[str]) -> dict[str, Any]:
    releases = _all_releases()
    chosen = sorted(set(selected))
    unknown = sorted(set(chosen) - set(releases))
    if unknown:
        raise ValueError(f"unknown NCCL release(s): {', '.join(unknown)}")
    release_matrix: list[dict[str, Any]] = []
    build_matrix: list[dict[str, Any]] = []
    for release in chosen:
        metadata = releases[release]
        release_matrix.append(
            {
                key: metadata[key]
                for key in (
                    "release",
                    "release_key",
                    "provider_id",
                    "provider_tag",
                    "release_alias",
                    "repository",
                )
            }
        )
        recipe = _load_json(RELEASES_ROOT / release / "recipe.json")
        for platform in recipe["builder"]["payload_platforms"]:
            item = dict(metadata)
            item.update(
                {
                    "platform": platform,
                    "arch": platform.rsplit("/", 1)[1],
                    "runner": RUNNERS[platform],
                }
            )
            build_matrix.append(item)
    return {
        "has_changes": bool(chosen),
        "build_matrix": {"include": build_matrix},
        "release_matrix": {"include": release_matrix},
    }


def _write_github_output(path: Path, plan: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"has_changes={str(plan['has_changes']).lower()}\n")
        for key in ("build_matrix", "release_matrix"):
            value = json.dumps(plan[key], separators=(",", ":"), sort_keys=True)
            output.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", help="select payloads changed after this Git ref")
    parser.add_argument("--head-ref", default="HEAD", help="Git ref compared with --base-ref")
    parser.add_argument("--changed-paths-file", type=Path, help="read changed paths from a file")
    parser.add_argument("--release", help="select one release, or 'all'")
    parser.add_argument("--github-output", type=Path, help="append GitHub Actions outputs")
    args = parser.parse_args()

    releases = _all_releases()
    selectors = sum(
        value is not None
        for value in (args.base_ref, args.changed_paths_file, args.release)
    )
    if selectors != 1:
        parser.error("select exactly one of --base-ref, --changed-paths-file, or --release")
    if args.release is not None:
        selected = set(releases) if args.release == "all" else {args.release}
    else:
        if args.changed_paths_file is not None:
            changed_paths = args.changed_paths_file.read_text(encoding="utf-8").splitlines()
        else:
            changed_paths = _changed_paths(args.base_ref, args.head_ref)
        selected = _select_changed(releases, changed_paths)
    plan = _make_plan(selected)
    if args.github_output is not None:
        _write_github_output(args.github_output, plan)
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
