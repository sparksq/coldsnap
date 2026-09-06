# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import io
import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime/engine"), str(ROOT / "runtime/shared")]
import coldsnap_service_runtime as runtime  # noqa: E402


def event(delta=None, *, finish=None):
    return ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}) + "\n\n").encode()


class StreamingAcceptanceTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(master_address="127.0.0.1", http_port=8000, model="test", prompt="Say OK", expected="OK", timeout=2)

    def test_one_request_times_reasoning_and_still_checks_complete_content(self):
        payload = event({"role": "assistant"}) + event({"content": ""}) + event({"reasoning_content": "Thinking"})
        payload += event({"content": "O"}) + event({"content": "K"}, finish="stop")
        payload += b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n\ndata: [DONE]\n\n'
        with patch.object(runtime, "_local_http_open", return_value=io.BytesIO(payload)) as send:
            response = runtime._infer(self.args())
        self.assertEqual(send.call_count, 1)
        request = json.loads(send.call_args.args[0].data)
        self.assertTrue(request["stream"])
        self.assertEqual(request["max_tokens"], 64)
        self.assertEqual(response["choices"][0]["message"]["content"], "OK")
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        timing = response["coldsnap_acceptance"]
        self.assertEqual(timing["first_token_field"], "reasoning_content")
        self.assertTrue(timing["response_validated"])
        self.assertLessEqual(timing["request_started_unix_ns"], timing["first_token_unix_ns"])
        self.assertLessEqual(timing["request_ttft_seconds"], timing["response_seconds"])

    def test_empty_wrong_truncated_error_and_oversize_streams_fail(self):
        for payload in (
            event(finish="stop") + b"data: [DONE]\n\n",
            event({"content": "wrong"}, finish="stop") + b"data: [DONE]\n\n",
            event({"content": "OK"}, finish="stop"),
            event({"content": "OK"}) + b"data: [DONE]\n\n",
            b'data: {"error": {"message": "failed"}}\n\n',
            b"data: " + b"x" * 65537,
        ):
            with self.subTest(payload=payload[:60]):
                with patch.object(runtime, "_local_http_open", return_value=io.BytesIO(payload)):
                    with self.assertRaises(RuntimeError):
                        runtime._infer(self.args())

    def test_failed_attempt_cannot_leak_timing_into_successful_retry(self):
        streams = [io.BytesIO(event({"content": "wrong"}, finish="stop") + b"data: [DONE]\n\n"),
                   io.BytesIO(event({"content": "OK"}, finish="stop") + b"data: [DONE]\n\n")]
        with patch.object(runtime, "_local_http_open", side_effect=streams):
            with self.assertRaises(RuntimeError):
                runtime._infer(self.args())
            retry_started = time.time_ns()
            response = runtime._infer(self.args())
        self.assertGreaterEqual(response["coldsnap_acceptance"]["first_token_unix_ns"], retry_started)

    def test_sse_comments_multiline_and_deadline(self):
        data = b': comment\r\ndata: {"choices":\r\ndata: []}\r\n\r\n'
        self.assertEqual(next(runtime._stream_events(io.BytesIO(data), time.perf_counter() + 1)), '{"choices":\n[]}')
        with self.assertRaises(TimeoutError):
            next(runtime._stream_events(io.BytesIO(data), time.perf_counter() - 1))

    def test_port_probe_matches_only_local_listeners_including_ipv6(self):
        header = "sl local_address rem_address st\n"
        for row, expected in (
            ("0: 00000000:1F40 00000000:0000 0A", True),
            ("0: 00000000:1F40 00000000:0000 06", False),
            ("0: 00000000:0040 00000000:1F40 0A", False),
        ):
            with patch.object(Path, "read_text", return_value=header + row):
                self.assertEqual(runtime._port_is_listening(8000), expected)
        with patch.object(Path, "read_text", side_effect=[FileNotFoundError(), header + "0: 00000000000000000000000000000000:1F40 00:0000 0A"]):
            self.assertTrue(runtime._port_is_listening(8000))

    def test_readiness_observes_independent_boundaries_and_no_inference(self):
        args = self.args()
        args.mode, args.rank, args.activation_state = "restore", 0, "running"
        response = SimpleNamespace(status=200)
        from contextlib import nullcontext
        with patch.object(runtime, "_port_is_listening", return_value=True), \
             patch.object(runtime, "_local_http_open", return_value=nullcontext(response)) as request:
            with runtime.StartupReadiness(args) as observer:
                for thread in observer.threads:
                    thread.join(timeout=1)
                timing = observer.snapshot()
            self.assertEqual(request.call_count, 1)
            self.assertTrue(request.call_args.args[0].endswith("/health"))
            self.assertLessEqual(timing["observer_started_unix_ns"], timing["port_open_unix_ns"])
            self.assertLessEqual(timing["observer_started_unix_ns"], timing["http_ready_unix_ns"])
        for mode, rank, state in (("capture", 0, "running"), ("restore", 1, "running"), ("restore", 0, "warm")):
            args.mode, args.rank, args.activation_state = mode, rank, state
            with runtime.StartupReadiness(args) as observer:
                self.assertEqual(observer.threads, [])

    def test_local_probe_bypasses_proxy_and_supports_ipv6(self):
        args = self.args()
        args.master_address = "::1"
        self.assertEqual(runtime._service_url(args, "/health"), "http://[::1]:8000/health")
        with patch.object(runtime.urllib.request, "build_opener") as build:
            runtime._local_http_open("http://localhost/health", timeout=1)
        self.assertEqual(build.call_args.args[0].proxies, {})
        build.return_value.open.assert_called_once_with("http://localhost/health", timeout=1)


if __name__ == "__main__":
    unittest.main()
