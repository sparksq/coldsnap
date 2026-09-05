# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "core"))
sys.path.insert(0, str(ROOT / "runtime" / "shared"))

import coldsnap_coord  # noqa: E402
from coldsnap_core.memory import (  # noqa: E402
    STANDARD_REGIONS,
    MemoryLayoutDescriptor,
    MemoryLayoutKind,
)


FIXTURE = json.loads(
    (ROOT / "test" / "fixtures" / "cross-language-v1.json").read_text(
        encoding="utf-8"
    )
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


class CrossLanguageContractFixtureTest(unittest.TestCase):
    def test_fixture_identity(self) -> None:
        self.assertEqual(FIXTURE["format"], 1)
        self.assertEqual(
            FIXTURE["kind"], "coldsnap-cross-language-contract-fixtures"
        )

    def test_canonical_json_and_manifest_digests(self) -> None:
        for item in FIXTURE["canonical_json"] + FIXTURE["manifests"]:
            with self.subTest(item["name"]):
                canonical = _canonical(item["value"])
                self.assertEqual(canonical, item["canonical"])
                self.assertEqual(
                    hashlib.sha256(canonical.encode("ascii")).hexdigest(),
                    item["sha256"],
                )

    def test_memory_layout_descriptor_digest(self) -> None:
        for item in FIXTURE["memory_layouts"]:
            with self.subTest(item["name"]):
                arguments = item["arguments"]
                descriptor = MemoryLayoutDescriptor.create(
                    provider=arguments["provider"],
                    provider_abi=arguments["provider_abi"],
                    region=STANDARD_REGIONS[arguments["region"]],
                    device=arguments["device"],
                    base_address=arguments["base_address"],
                    logical_bytes=arguments["logical_bytes"],
                    allocation_granularity=arguments["allocation_granularity"],
                    mapping_granularity=arguments["mapping_granularity"],
                    layout_kind=MemoryLayoutKind(arguments["layout_kind"]),
                    groups=tuple(arguments["groups"]),
                    extent_count=arguments["extent_count"],
                    transport_registration_mode=arguments[
                        "transport_registration_mode"
                    ],
                    extension=arguments["extension"],
                )
                self.assertEqual(descriptor.extension_json, item["extension_json"])
                self.assertEqual(descriptor.layout_digest, item["layout_digest"])
                fields_digest = hashlib.sha256(
                    _canonical(item["fields"]).encode("ascii")
                ).hexdigest()
                self.assertEqual("sha256:" + fields_digest, item["layout_digest"])

    def test_coordinator_frames(self) -> None:
        for item in FIXTURE["coordinator_frames"]:
            with self.subTest(item["name"]):
                if "request" in item:
                    request = item["request"]
                    endpoint = coldsnap_coord.Endpoint(
                        "127.0.0.1:1",
                        request["key_utf8"].split("/", 1)[0],
                        request["token_utf8"],
                    )
                    key = request["key_utf8"].split("/", 1)[1]
                    encoded = coldsnap_coord._encode_request(
                        endpoint,
                        request["operation"],
                        key,
                        request["value_utf8"].encode("utf-8"),
                        request["timeout_ms"] / 1000,
                    )
                else:
                    response = item["response"]
                    value = response["value_utf8"].encode("utf-8")
                    encoded = coldsnap_coord._RESPONSE.pack(
                        coldsnap_coord.MAGIC,
                        coldsnap_coord.VERSION,
                        response["status"],
                        0,
                        len(value),
                    ) + value
                self.assertEqual(encoded.hex(), item["hex"])


if __name__ == "__main__":
    unittest.main()
