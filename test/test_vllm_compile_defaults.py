# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""Compilation defaults must remain compatible with older vLLM images."""

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from coldsnap_startup_plan import install_aot_compilation_defaults  # noqa: E402


STANDALONE = "VLLM_USE_STANDALONE_COMPILE"
AOT = "VLLM_USE_AOT_COMPILE"
MEGA = "VLLM_USE_MEGA_AOT_ARTIFACT"
DISABLE_CACHE = "VLLM_DISABLE_COMPILE_CACHE"


class CompilationDefaultsTest(unittest.TestCase):
    def configure(self, supported, environment=None, registered=None):
        names = registered if registered is not None else (STANDALONE, AOT, MEGA)
        getters = {
            name: (lambda key=name: os.environ.get(key, "0") == "1")
            for name in (*names, DISABLE_CACHE)
        }
        vllm = ModuleType("vllm")
        vllm.envs = SimpleNamespace(environment_variables=getters)
        torch_utils = ModuleType("vllm.utils.torch_utils")
        torch_utils.is_torch_equal_or_newer = lambda version: version in supported
        logger = ModuleType("vllm.logger")
        logger.init_logger = Mock(return_value=Mock())
        with patch.dict(os.environ, environment or {}, clear=True), patch.dict(
            sys.modules,
            {"vllm": vllm, "vllm.utils.torch_utils": torch_utils, "vllm.logger": logger},
        ):
            install_aot_compilation_defaults()
            first = dict(os.environ)
            install_aot_compilation_defaults()
            self.assertEqual(dict(os.environ), first)
            return first

    def test_supported_features_are_enabled_by_torch_version(self):
        floors = ("2.9.0", "2.10.0", "2.12.0.dev")
        flags = (STANDALONE, AOT, MEGA)
        for count in range(4):
            with self.subTest(count=count):
                self.assertEqual(
                    self.configure(floors[:count]),
                    {name: "1" for name in flags[:count]},
                )

    def test_only_registered_vllm_features_are_enabled(self):
        self.assertEqual(self.configure(("2.9.0", "2.10.0", "2.12.0.dev"), registered=()), {})
        self.assertEqual(
            self.configure(("2.9.0", "2.10.0", "2.12.0.dev"), registered=(AOT,)),
            {AOT: "1"},
        )

    def test_explicit_opt_outs_preserve_dependencies(self):
        floors = ("2.9.0", "2.10.0", "2.12.0.dev")
        self.assertEqual(self.configure(floors, {AOT: "0"}), {AOT: "0", STANDALONE: "1"})
        self.assertEqual(self.configure(floors, {STANDALONE: "0"}), {AOT: "1", STANDALONE: "0"})
        self.assertEqual(
            self.configure(floors, {MEGA: "0"}),
            {AOT: "1", STANDALONE: "1", MEGA: "0"},
        )

    def test_disabled_cache_preserves_recipe_settings(self):
        environment = {DISABLE_CACHE: "1", AOT: "0"}
        self.assertEqual(self.configure(("2.9.0", "2.10.0", "2.12.0.dev"), environment), environment)

    def test_explicit_opt_in_on_older_toolchain_is_not_overridden(self):
        self.assertEqual(self.configure((), {AOT: "1"}), {AOT: "1"})

    def test_missing_version_helper_leaves_older_images_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"vllm": None}):
            install_aot_compilation_defaults()
            self.assertEqual(dict(os.environ), {})
