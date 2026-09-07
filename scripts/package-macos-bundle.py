#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Package native macOS controllers into a deterministic Darwin OCI layout.

No Linux image or Docker daemon is involved. The resulting manifest can share
one release index with the existing Linux binary images.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

BINARIES = ("coldsnap", "coldsnap-vllm-adapter", "coldsnap-sglang-adapter")
OCI = "application/vnd.oci.image."


def encode(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def package(root: Path, output: Path, *, version: str, commit: str, arch: str, source: Path) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or arch not in {"amd64", "arm64"}:
        raise ValueError("a full source commit and supported macOS architecture are required")
    output.mkdir(parents=True, exist_ok=False)
    blobs = output / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def blob(data, media_type):
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return {"mediaType": media_type, "digest": "sha256:" + digest, "size": len(data)}

    payloads = {}
    for name in BINARIES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("bundle executables must be regular files")
        payloads[name] = path.read_bytes()
    manifest = {
        "format": 1, "kind": "coldsnap-controller-binary-bundle", "version": version,
        "commit": commit, "platform": "darwin-" + arch,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["manifest.json"] = encode(manifest)
    (root / "manifest.json").write_bytes(payloads["manifest.json"])
    files = [("opt/coldsnap/bin/" + name, data, 0o755 if name in BINARIES else 0o644)
             for name, data in sorted(payloads.items())]
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        files.append(("usr/share/licenses/coldsnap/" + name, (source / name).read_bytes(), 0o644))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data, mode in files:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), mode
            tar.addfile(member, io.BytesIO(data))
    layer = archive.getvalue()
    config = {
        "os": "darwin", "architecture": arch,
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(layer).hexdigest()]},
        "config": {"Labels": {"org.opencontainers.image.version": version, "org.opencontainers.image.revision": commit}},
    }
    image = {
        "schemaVersion": 2, "mediaType": OCI + "manifest.v1+json",
        "config": blob(encode(config), OCI + "config.v1+json"),
        "layers": [blob(gzip.compress(layer, mtime=0), OCI + "layer.v1.tar+gzip")],
    }
    descriptor = blob(encode(image), OCI + "manifest.v1+json")
    descriptor["platform"] = {"os": "darwin", "architecture": arch}
    descriptor["annotations"] = {"org.opencontainers.image.ref.name": "bundle"}
    (output / "index.json").write_bytes(encode({"schemaVersion": 2, "mediaType": OCI + "index.v1+json", "manifests": [descriptor]}))
    (output / "oci-layout").write_bytes(encode({"imageLayoutVersion": "1.0.0"}))
    return descriptor["digest"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--arch", choices=("amd64", "arm64"), required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(package(args.root, args.output, version=args.version, commit=args.commit, arch=args.arch, source=args.source))
