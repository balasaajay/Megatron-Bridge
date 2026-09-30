# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage only an exact-source bootstrap, instrumented entrypoint, and recorder."""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import nemo_run as run
from nemo_run.core.packaging.base import Packager
from nemo_run.run.torchx_backend.schedulers.api import EXECUTOR_MAPPING


FROZEN_ENTRYPOINT_HASHES = {
    "bootstrap.py": "6917db72f631947d0a299356a134d88384981f313f6b0061b233e66999cf0805",
    "run_recipe.py": "26633e8f61d7adbb5b92a8fba58bd0415d0d66b713da9ae045e62ce0f5c9acd5",
}
BOOTSTRAP_IMPORT = "from argument_parser import parse_cli_args\n"
BOOTSTRAP_PRELUDE = """# Keep the staged sibling first and resolve all framework imports from the image.
_gc_image_paths = ["/opt/Megatron-Bridge/scripts/performance", "/opt/Megatron-Bridge/src"]
sys.path[1:1] = _gc_image_paths
_gc_inherited_path = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(_gc_image_paths + ([_gc_inherited_path] if _gc_inherited_path else []))

from argument_parser import parse_cli_args
"""
PRETRAIN_CALL = "        pretrain(config=recipe, forward_step_func=forward_step)\n"
INSTRUMENTED_PRETRAIN_CALL = """        from gc_timing_recorder import EVENT_CAPACITY, LONG_EVENT_NS, GCTimingCallback

        if recipe.train.manual_gc:
            raise ValueError("GC timing requires the frozen automatic-GC workload policy.")
        gc_timing = GCTimingCallback(capacity=EVENT_CAPACITY, long_event_ns=LONG_EVENT_NS)
        try:
            pretrain(config=recipe, forward_step_func=forward_step, callbacks=[gc_timing])
        finally:
            gc_timing.close()
"""


def stage_gc_timing(*, destination: Path, performance_dir: Path) -> Path:
    """Create or verify a three-file bundle without overlaying the trainer library.

    Input entrypoints must match Bridge 8f160254 byte-for-byte. Existing staging
    directories are accepted only if their complete contents match this bundle.
    Nothing imports or modifies the image parser, library, or MCore source.
    """
    payloads = {}
    for filename, expected_hash in FROZEN_ENTRYPOINT_HASHES.items():
        content = (performance_dir / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"GC diagnostic requires the exact frozen 8f160254 {filename}.")
        payloads[filename] = content
    original = payloads["run_recipe.py"].decode()
    if original.count(PRETRAIN_CALL) != 1:
        raise ValueError("Expected exactly one frozen pretrain call for GC injection.")
    payloads["run_recipe.py"] = original.replace(PRETRAIN_CALL, INSTRUMENTED_PRETRAIN_CALL).encode()
    bootstrap = payloads["bootstrap.py"].decode()
    if bootstrap.count(BOOTSTRAP_IMPORT) != 1:
        raise ValueError("Expected exactly one frozen bootstrap parser import.")
    payloads["bootstrap.py"] = bootstrap.replace(BOOTSTRAP_IMPORT, BOOTSTRAP_PRELUDE).encode()
    payloads["gc_timing_recorder.py"] = (performance_dir / "utils" / "gc_timing_recorder.py").read_bytes()
    if destination.is_symlink():
        raise ValueError("GC staging destination must not be a symlink.")
    if destination.exists():
        expected = {"gc_probe", *(f"gc_probe/{name}" for name in payloads)}
        actual = {str(path.relative_to(destination)) for path in destination.rglob("*")}
        if actual != expected or any(path.is_symlink() for path in destination.rglob("*")):
            raise ValueError("Existing GC stage must contain only the exact diagnostic bundle.")
        if any((destination / "gc_probe" / name).read_bytes() != content for name, content in payloads.items()):
            raise ValueError("Existing GC stage differs from the diagnostic bundle.")
    else:
        probe = destination / "gc_probe"
        probe.mkdir(parents=True)
        for name, content in payloads.items():
            (probe / name).write_bytes(content)
    return destination.resolve()


@dataclass(kw_only=True)
class GCTimingKubeflowExecutor(run.KubeflowExecutor):
    """Stage the verified probe after experiment preparation, before native upload."""

    gc_staging_dir: str | None = None
    gc_performance_dir: str | None = None

    def materialize_launch_script(self, cmd: list[str], max_retries: int = 0) -> None:
        """Use the assigned PVC path; the image's /nemo_run is a real directory."""
        bootstrap = "/nemo_run/gc_probe/bootstrap.py"
        if (
            cmd.count(bootstrap) != 1
            or not self.workdir_pvc
            or not getattr(self, "experiment_id", None)
            or not getattr(self, "job_name", None)
        ):
            raise ValueError("GC timing requires one unwrapped bootstrap token and an assigned PVC path.")
        resolved = f"{self.code_dir}/gc_probe/bootstrap.py"
        super().materialize_launch_script(
            [resolved if token == bootstrap else token for token in cmd], max_retries=max_retries
        )

    def package(self, packager: Packager, job_name: str) -> None:
        """Copy only owned probe files; preserve ordinary Kubeflow upload/cleanup."""
        if (
            self.workdir_local_path
            or not self.workdir_pvc
            or not self.job_dir
            or not Path(self.job_dir).is_dir()
            or not self.gc_staging_dir
            or not self.gc_performance_dir
        ):
            raise ValueError("GC timing requires a prepared job directory, a workdir PVC, and no source overlay.")
        source = stage_gc_timing(destination=Path(self.gc_staging_dir), performance_dir=Path(self.gc_performance_dir))
        probe_source = source / "gc_probe"
        probe_target = Path(self.job_dir) / "gc_probe"
        if probe_target.exists() or probe_target.is_symlink():
            expected = {path.name: path.read_bytes() for path in probe_source.iterdir()}
            if (
                probe_target.is_symlink()
                or not probe_target.is_dir()
                or {path.name for path in probe_target.iterdir()} != set(expected)
                or any(
                    path.is_symlink() or not path.is_file() or path.read_bytes() != expected[path.name]
                    for path in probe_target.iterdir()
                )
            ):
                raise ValueError("Assigned job directory contains a different GC timing bundle.")
        else:
            shutil.copytree(probe_source, probe_target)
        super().package(packager, job_name)


# Frozen NeMo-Run uses exact types in both dispatch and its parallel-DAG gate.
# Preserve the base/reverse mapping and ordinary attached-only Kubeflow behavior.
EXECUTOR_MAPPING[GCTimingKubeflowExecutor] = EXECUTOR_MAPPING[run.KubeflowExecutor]
if GCTimingKubeflowExecutor not in run.Experiment._PARALLEL_SUPPORTED_EXECUTORS:
    run.Experiment._PARALLEL_SUPPORTED_EXECUTORS += (GCTimingKubeflowExecutor,)
