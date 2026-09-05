# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only

"""SGLang adapter over the shared ColdSnap region policy and TMS VMM."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coldsnap_core.memory import CUDA_GRAPH, WEIGHTS

from .settings import Settings


_TMS_PRELOAD_ENV = "COLDSNAP_SGLANG_TMS_PRELOAD"


@dataclass(frozen=True)
class _RegisteredModel:
    model: Any
    identity: dict[str, Any]
    artifact: Any


class ColdSnapMemorySaverAdapter:
    """Preserve SGLang's adapter shape while making policies explicit."""

    def __init__(self, settings: Settings) -> None:
        try:
            import torch_memory_saver
        except ImportError as error:
            raise RuntimeError(
                "ColdSnap SGLang requires torch_memory_saver for stable CUDA VMM regions"
            ) from error
        self._module = torch_memory_saver
        self._saver = torch_memory_saver.torch_memory_saver
        self._settings = settings
        self._disk_configured = False
        self._models: list[_RegisteredModel] = []
        parameters = inspect.signature(self._saver.region).parameters
        if settings.live_backing == "disk" and "enable_disk_backup" not in parameters:
            raise RuntimeError(
                "installed torch_memory_saver lacks enable_disk_backup; use the ColdSnap TMS fork or COLDSNAP_LIVE_BACKING=cpu"
            )

    @property
    def enabled(self) -> bool:
        return True

    def check_validity(self, caller_name: str) -> None:
        del caller_name

    @staticmethod
    def _preload_entries(value: str | None) -> list[str]:
        return [entry for entry in (value or "").split(":") if entry]

    def _tms_preload_path(self) -> str:
        active = self._preload_entries(os.environ.get("LD_PRELOAD"))
        declared = os.environ.get(_TMS_PRELOAD_ENV)
        if declared:
            if declared not in active:
                raise RuntimeError("ColdSnap SGLang memory-saver preload is absent from LD_PRELOAD")
            return declared
        matches = [entry for entry in active if "torch_memory_saver" in Path(entry).name]
        if len(matches) != 1:
            raise RuntimeError("ColdSnap SGLang requires exactly one torch_memory_saver preload")
        return matches[0]

    @contextmanager
    def _tms_library_environment(self):
        # torch_memory_saver's preload hook is already mapped by exec, but its
        # Python wrapper also treats LD_PRELOAD as the singular filename to
        # pass to ctypes.CDLL. Present that narrow view only while entering its
        # API; the complete ColdSnap preload remains the process-start view.
        previous = os.environ.get("LD_PRELOAD")
        os.environ["LD_PRELOAD"] = self._tms_preload_path()
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("LD_PRELOAD", None)
            else:
                os.environ["LD_PRELOAD"] = previous

    @contextmanager
    def configure_subprocess(self):
        inherited = os.environ.get("LD_PRELOAD")
        with self._module.configure_subprocess():
            memory_saver = os.environ.get("LD_PRELOAD")
            tms_entries = self._preload_entries(memory_saver)
            if len(tms_entries) != 1 or "torch_memory_saver" not in Path(tms_entries[0]).name:
                raise RuntimeError("SGLang torch_memory_saver returned an invalid preload path")
            entries: list[str] = []
            for entry in self._preload_entries(inherited) + self._preload_entries(memory_saver):
                if entry not in entries:
                    entries.append(entry)
            previous = os.environ.get("LD_PRELOAD")
            previous_tms = os.environ.get(_TMS_PRELOAD_ENV)
            if entries:
                os.environ["LD_PRELOAD"] = ":".join(entries)
            else:
                os.environ.pop("LD_PRELOAD", None)
            os.environ[_TMS_PRELOAD_ENV] = tms_entries[0]
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = previous
                if previous_tms is None:
                    os.environ.pop(_TMS_PRELOAD_ENV, None)
                else:
                    os.environ[_TMS_PRELOAD_ENV] = previous_tms

    def _ensure_disk_dir(self) -> None:
        if self._disk_configured or self._settings.live_backing != "disk":
            return
        assert self._settings.live_root is not None
        path = self._settings.live_root / f"pid-{os.getpid()}"
        path.mkdir(parents=True, exist_ok=True)
        self._saver.set_disk_backup_dir(str(path))
        self._disk_configured = True

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        del enable_cpu_backup
        kwargs: dict[str, Any] = {"tag": tag, "enable_cpu_backup": False}
        if tag == WEIGHTS.name:
            self._ensure_disk_dir()
            kwargs["enable_cpu_backup"] = self._settings.live_backing == "cpu"
            if self._settings.live_backing == "disk":
                kwargs["enable_disk_backup"] = True
        with self._tms_library_environment():
            with self._saver.region(**kwargs):
                yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        kwargs["tag"] = CUDA_GRAPH.name
        kwargs["enable_cpu_backup"] = False
        with self._tms_library_environment():
            with self._saver.cuda_graph(**kwargs):
                yield

    @contextmanager
    def disable(self):
        with self._tms_library_environment():
            with self._saver.disable():
                yield

    def pause(self, tag: str):
        if tag == WEIGHTS.name:
            self._ensure_disk_dir()
        with self._tms_library_environment():
            return self._saver.pause(tag=tag)

    def resume(self, tag: str):
        with self._tms_library_environment():
            result = self._saver.resume(tag=tag)
        if tag == WEIGHTS.name:
            self._restore_registered_models()
        return result

    def register_model(self, model: Any, identity: dict[str, Any], artifact: Any) -> None:
        if any(item.model is model for item in self._models):
            return
        self._models.append(_RegisteredModel(model=model, identity=identity, artifact=artifact))

    def restore_startup_model(self, model: Any, identity: dict[str, Any], artifact: Any) -> None:
        """Hydrate one freshly constructed model before startup warmup/graphs."""
        if self._settings.startup_provider != "native":
            return
        matches = [item for item in self._models if item.model is model]
        if len(matches) != 1:
            raise RuntimeError("ColdSnap SGLang startup model was not registered exactly once")
        registered = matches[0]
        if registered.identity != identity or registered.artifact is not artifact:
            raise RuntimeError("ColdSnap SGLang startup model identity changed")
        self._restore_models((registered,))

    def _worker_payload_root(self) -> Path | None:
        root = self._settings.process_artifact_root
        if root is None:
            return None
        from coldsnap_core.topology import worker_id

        return root / "hydration" / worker_id()

    @staticmethod
    def _selected_provider(root: Path | None) -> str:
        if root is None:
            raise RuntimeError("ColdSnap SGLang process artifact root is unavailable")
        selector = root / "activation-provider"
        try:
            value = selector.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise RuntimeError(
                "ColdSnap SGLang activation provider was not selected by the controller"
            ) from error
        if value not in {"native", "recovery"}:
            raise RuntimeError(f"invalid ColdSnap SGLang weight provider {value!r}")
        return value

    def _restore_registered_models(self) -> None:
        if not self._models:
            raise RuntimeError("ColdSnap SGLang resumed weights without registered models")
        root = self._worker_payload_root()
        provider = self._selected_provider(root)
        if provider == "recovery":
            # SGLang's controller calls update_weights_from_disk after the
            # address-stable allocations have been remapped.
            return

        self._restore_models(tuple(self._models))

    def _restore_models(self, models: tuple[_RegisteredModel, ...]) -> None:
        if not models:
            raise RuntimeError("ColdSnap SGLang native restore has no models")
        root = self._worker_payload_root()

        assert root is not None
        index = json.loads((root / "native-manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(index, dict)
            or index.get("format") != 1
            or index.get("kind") != "coldsnap-sglang-model-payload"
            or not isinstance(index.get("components"), list)
        ):
            raise RuntimeError("ColdSnap SGLang native payload index is invalid")
        pack = root / "model-weights.pack"
        if not pack.is_file():
            raise RuntimeError("ColdSnap SGLang native model payload is unavailable")

        by_artifact = {
            str(component.get("artifact")): component
            for component in (index or {}).get("components", [])
            if isinstance(component, dict)
        }
        semantic_root = self._settings.artifact_root
        for item in models:
            blob_path = None
            blob_offset = 0
            if semantic_root is None:
                raise RuntimeError("ColdSnap SGLang semantic root is unavailable")
            key = item.artifact.root.relative_to(semantic_root).as_posix()
            component = by_artifact.get(key)
            if component is None or not isinstance(component.get("offset"), int):
                raise RuntimeError(f"ColdSnap SGLang native payload lacks component {key!r}")
            blob_path = pack
            blob_offset = int(component["offset"])
            metrics = item.artifact.restore(
                item.model,
                identity=item.identity,
                chunk_bytes=256 * 1024**2,
                queue_depth=4,
                preverified=self._settings.preverified,
                blob_path=blob_path,
                blob_offset=blob_offset,
            )
            # Keep this module independent of SGLang's logging initialization.
            import logging

            logging.getLogger(__name__).info(
                "ColdSnap SGLang live restore: backend=%s bytes=%d storages=%d elapsed=%.3f s",
                metrics.backend,
                metrics.bytes,
                metrics.storages,
                metrics.seconds,
            )


def create_adapter(settings: Settings) -> ColdSnapMemorySaverAdapter:
    return ColdSnapMemorySaverAdapter(settings)
