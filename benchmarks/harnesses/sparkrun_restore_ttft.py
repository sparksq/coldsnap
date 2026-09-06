#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Measure Docker StartedAt through the first non-empty streamed model token."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement", choices=("external", "rank0"), default="external")
    parser.add_argument("--rank0-source", choices=("acceptance", "inference"), default="acceptance",
                        help="Reuse ColdSnap acceptance or probe a normal launch through Sparkrun")
    parser.add_argument(
        "--ready-file",
        type=Path,
        help=(
            "Write a synchronization marker after pre-existing containers have "
            "been snapshotted, before waiting for the measured launch"
        ),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Select the newest matching container even when observation starts after launch",
    )
    return parser.parse_args()


def _remote(host: str, *arguments: str) -> str:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            shlex.join(arguments),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _rfc3339_ns(value: str) -> int:
    date, _, fractional = value.removesuffix("Z").partition(".")
    seconds = int(datetime.strptime(date, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp())
    return seconds * 1_000_000_000 + int((fractional + "000000000")[:9])


def _container_rows(args: argparse.Namespace) -> list[tuple[str, str]]:
    rows = _remote(
        args.docker_host,
        "docker",
        "ps",
        "-a",
        "--filter",
        f"label=sparkrun.intent_id={args.intent_id}",
        "--filter",
        "label=sparkrun.rank=0",
        "--format",
        "{{.ID}} {{.Names}}",
    ).splitlines()
    return [tuple(row.split(maxsplit=1)) for row in rows]


def _wait_container(
    args: argparse.Namespace, existing_container_ids: set[str]
) -> tuple[str, str, int]:
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        rows = _container_rows(args)
        candidates: list[tuple[int, str, str, str]] = []
        for container_id, name in rows:
            if container_id in existing_container_ids:
                continue
            timestamp = _remote(
                args.docker_host,
                "docker",
                "inspect",
                "--format",
                "{{.State.StartedAt}}",
                container_id,
            )
            timestamp_ns = _rfc3339_ns(timestamp)
            candidates.append((timestamp_ns, container_id, name, timestamp))
        if candidates:
            timestamp_ns, container_id, name, timestamp = max(candidates)
            return container_id, name, timestamp_ns
        time.sleep(0.1)
    raise TimeoutError("new rank-0 restore container did not start")


def _wait_health(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    url = args.api_base.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return time.time_ns()
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    raise TimeoutError("health endpoint did not become ready")


def _stream(args: argparse.Namespace) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            response = urllib.request.urlopen(request, timeout=args.timeout)
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
            continue
        first_token_ns: int | None = None
        first_token_field = ""
        first_token_text = ""
        pieces: dict[str, list[str]] = {
            "content": [],
            "reasoning": [],
            "reasoning_content": [],
        }
        usage: dict[str, object] | None = None
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                for field in pieces:
                    text = delta.get(field)
                    if isinstance(text, str) and text:
                        pieces[field].append(text)
                        if first_token_ns is None:
                            first_token_ns = time.time_ns()
                            first_token_field = field
                            first_token_text = text
        if first_token_ns is not None:
            return {
                "first_token_observed_unix_ns": first_token_ns,
                "first_token_field": first_token_field,
                "first_token_text": first_token_text,
                "content": "".join(pieces["content"]),
                "reasoning": "".join(pieces["reasoning"]),
                "reasoning_content": "".join(pieces["reasoning_content"]),
                "usage": usage,
            }
        # A restored vLLM API process can answer while its engine remains in
        # sleep mode. That response is a successful empty stream, not TTFT.
        time.sleep(0.05)
    raise TimeoutError("stream did not produce a non-empty token")


def _rank0(args: argparse.Namespace, container_id: str, started_ns: int) -> dict[str, object]:
    """Reuse the built-in probe; never issue another ColdSnap inference request."""
    if args.rank0_source == "inference":
        from urllib.parse import urlsplit
        from sparkrun.orchestration.startup import run_probe

        url = urlsplit(args.api_base)
        observation = run_probe(args.docker_host, {
            "container": container_id, "address": url.hostname, "port": url.port or 8000,
            "port_timeout_s": args.timeout, "health_timeout_s": args.timeout,
            "inference_timeout_s": args.timeout, "prompt": args.prompt, "expected": args.expected,
        })
        if observation["container_started_unix_ns"] != started_ns:
            raise RuntimeError("measured container restarted during observation")
        observation["exact_response"] = observation.get("response_validated") is True
    else:
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                payload = _remote(args.docker_host, "docker", "exec", container_id,
                                  "cat", "/opt/coldsnap/capsule/restore-ready.json")
                report = json.loads(payload)
                break
            except subprocess.CalledProcessError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank-0 acceptance receipt was not published") from None
                running = _remote(args.docker_host, "docker", "inspect", "--format", "{{.State.Running}}", container_id)
                if running != "true":
                    raise RuntimeError("rank-0 container exited before acceptance") from None
                time.sleep(0.1)
        response = report.get("post_restore_response") or {}
        acceptance = response.get("coldsnap_acceptance") or {}
        if report.get("rank") != 0 or acceptance.get("format") != 1 or acceptance.get("observer") != "rank0" or acceptance.get("measurement") != "rank0-acceptance-v1" or acceptance.get("response_validated") is not True:
            raise RuntimeError("runtime has no validated rank-0 streaming acceptance; upgrade ColdSnap")
        if acceptance.get("prompt_sha256") != hashlib.sha256(args.prompt.encode()).hexdigest():
            raise RuntimeError("acceptance prompt does not match the qualification prompt")
        if acceptance.get("max_tokens") != 64 or acceptance.get("temperature") != 0:
            raise RuntimeError("acceptance sampling settings do not match rank-0 qualification")
        timestamp = _remote(args.docker_host, "docker", "inspect", "--format", "{{.State.StartedAt}}", container_id)
        if _rfc3339_ns(timestamp) != started_ns:
            raise RuntimeError("measured container restarted during observation")
        exact = response["choices"][0]["message"]["content"].strip() == args.expected
        observation = {
            **acceptance, **acceptance.get("readiness", {}),
            "container_id": container_id, "container_started_unix_ns": started_ns,
            "exact_response": exact, "inference_ready": True,
            "content": response["choices"][0]["message"]["content"],
        }
    first = observation["first_token_unix_ns"]
    if type(first) is not int or first < started_ns or observation.get("first_token_field") not in ("content", "reasoning", "reasoning_content") or not observation["exact_response"]:
        raise RuntimeError("invalid or unsuccessful rank-0 observation")
    observation.update(format=1, kind="coldsnap-sparkrun-rank0-startup-observation",
                       first_token_observed_unix_ns=first,
                       container_to_first_token_seconds=(first - started_ns) / 1e9)
    for source, target in (("port_open_unix_ns", "container_to_port_open_seconds"),
                           ("http_ready_unix_ns", "container_to_health_seconds")):
        timestamp = observation.get(source)
        if timestamp is not None:
            if type(timestamp) is not int or timestamp < started_ns:
                raise RuntimeError("invalid rank-0 readiness timestamp")
            observation[target] = (timestamp - started_ns) / 1e9
    return observation


def main() -> int:
    args = _arguments()
    observer_started_ns = time.time_ns()
    existing_container_ids = (
        set()
        if args.include_existing
        else {container_id for container_id, _ in _container_rows(args)}
    )
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(
            json.dumps(
                {
                    "observer_started_unix_ns": observer_started_ns,
                    "existing_container_ids": sorted(existing_container_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    container_id, container_name, container_started_ns = _wait_container(
        args, existing_container_ids
    )
    if args.measurement == "rank0":
        result = _rank0(args, container_id, container_started_ns)
        result["container_name"] = container_name
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    health_ns = _wait_health(args)
    stream = _stream(args)
    first_token_ns = int(stream["first_token_observed_unix_ns"])
    exact_response = str(stream["content"]).strip() == args.expected
    result = {
        "format": 1,
        "kind": "coldsnap-sparkrun-restore-ttft-observation",
        "measurement": "external-stream-v1",
        "observer_started_unix_ns": observer_started_ns,
        "container_id": container_id,
        "container_name": container_name,
        "container_started_at": datetime.fromtimestamp(
            container_started_ns / 1_000_000_000, tz=UTC
        ).isoformat(),
        "container_started_unix_ns": container_started_ns,
        "health_observed_unix_ns": health_ns,
        "container_to_health_seconds": (health_ns - container_started_ns) / 1e9,
        "container_to_first_token_seconds": (first_token_ns - container_started_ns) / 1e9,
        "health_to_first_token_seconds": (first_token_ns - health_ns) / 1e9,
        "expected": args.expected,
        "exact_response": exact_response,
        **stream,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not exact_response:
        raise RuntimeError("streamed content did not match the expected exact response")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
