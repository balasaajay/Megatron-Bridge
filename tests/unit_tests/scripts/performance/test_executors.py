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

"""Tests for scripts/performance/utils/executors.py — container_env on SlurmExecutor."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


# scripts/performance is not an installed package; add it to sys.path so we
# can import ``utils.executors`` the same way the scripts themselves do.
_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from setup_experiment import _build_nemorun_script
    from utils import executors as executors_module
    from utils.executors import (
        KUBEFLOW_NUMA_BINDING_ENV,
        OFFLINE_BENCHMARK_ENV_VARS,
        DiagnosticKubeflowExecutor,
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        kubeflow_executor,
        slurm_executor,
    )


RECIPE_PROCESS_ENV_NAMES = {
    "NCCL_GRAPH_REGISTER",
    "NCCL_NVLS_ENABLE",
    "NVTE_NORM_BWD_USE_CUDNN",
    "NVTE_NORM_FWD_USE_CUDNN",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_HIGH_PRIORITY",
}


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_container_env_includes_offline_benchmark_vars(tmp_path):
    """Offline benchmark defaults must override matching container values."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )
    assert executor.container_env is not None, "container_env is None — was the field removed from the executor?"
    missing = set(OFFLINE_BENCHMARK_ENV_VARS) - set(executor.container_env)
    assert not missing, f"Offline benchmark vars missing from container_env: {missing}"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_custom_env_vars_in_container_env(tmp_path):
    """Vars passed via custom_env_vars must also appear in container_env."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
        custom_env_vars={"MY_CUSTOM_VAR": "1"},
    )
    assert "MY_CUSTOM_VAR" in executor.container_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_numa_binding_wraps_each_torchrun_worker():
    """The opt-in Kubeflow launcher must resolve and bind each worker's local GPU."""
    assert _kubeflow_numa_binding_enabled({KUBEFLOW_NUMA_BINDING_ENV: "1"})
    task = nemo_run.Script(
        path="train.py",
        entrypoint="python",
        args=["--steps", "10"],
        env={"PYTHONPATH": "/workspace:$PYTHONPATH"},
        metadata={"test": "value"},
    )
    wrapper = _kubeflow_numa_binding_script(task)
    assert 'nvidia-smi -i "$LOCAL_RANK"' in wrapper.inline
    assert "head -n1" not in wrapper.inline
    assert 'NUMA_FILE="/sys/bus/pci/devices/$PCI_BUS/numa_node"' in wrapper.inline
    assert (
        'exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" python train.py --steps 10' in wrapper.inline
    )
    assert wrapper.env == task.env
    assert wrapper.env is not task.env
    assert wrapper.metadata == task.metadata
    assert wrapper.metadata is not task.metadata


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_build_nemorun_script_wraps_only_enabled_kubeflow_tasks():
    """The setup helper must preserve task env while gating the Kubeflow wrapper."""
    kwargs = {
        "script_path": "/opt/Megatron-Bridge/scripts/performance/run_recipe.py",
        "script_dir": "/opt/Megatron-Bridge/scripts/performance",
        "args": ["--steps", "10"],
    }
    enabled = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace="nemo-ci",
        custom_env_vars={KUBEFLOW_NUMA_BINDING_ENV: "1"},
    )
    disabled = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace="nemo-ci",
        custom_env_vars={},
    )
    non_kubeflow = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace=None,
        custom_env_vars={KUBEFLOW_NUMA_BINDING_ENV: "1"},
    )

    expected_env = {"PYTHONPATH": "/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:$PYTHONPATH"}
    assert enabled.inline
    assert enabled.env == expected_env
    assert not disabled.inline
    assert disabled.env == expected_env
    assert not non_kubeflow.inline
    assert non_kubeflow.env == expected_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_numa_binding_is_disabled_by_default():
    """Normal Kubeflow jobs must retain the unmodified Torchrun launcher."""
    assert not _kubeflow_numa_binding_enabled({})
    assert not _kubeflow_numa_binding_enabled({KUBEFLOW_NUMA_BINDING_ENV: "0"})


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_executor_never_supplies_recipe_process_defaults(tmp_path):
    """Flat performance and model recipes supply process settings without executor shadowing."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )

    assert RECIPE_PROCESS_ENV_NAMES.isdisjoint(executor.env_vars)
    assert "TRANSFORMERS_OFFLINE" in executor.env_vars


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_recipe_env_vars_are_exported_and_forced_into_container(tmp_path):
    """Launcher-resolved recipe vars must reach Slurm tasks before Python starts."""
    recipe_env_vars = {
        "TORCHINDUCTOR_WORKER_START": "fork",
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "64",
    }
    executor = slurm_executor(
        gpu="gb200",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=2,
        num_gpus_per_node=4,
        custom_env_vars=recipe_env_vars,
    )

    assert executor.env_vars.items() >= recipe_env_vars.items()
    assert set(recipe_env_vars) <= set(executor.container_env or [])


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_vr200_slurm_executor_uses_two_gpus_per_numa_node(tmp_path):
    executor = slurm_executor(
        gpu="vr200",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=4,
    )

    assert "SLURM_LOCALID/2" in executor.launcher.template_vars["pre_cmds"]


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_recipe_env_vars_are_added_to_kubeflow_trainer_environment(monkeypatch):
    """Kubeflow workers should inherit launcher-resolved recipe variables."""
    monkeypatch.setattr(
        executors_module,
        "DiagnosticKubeflowExecutor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    recipe_env_vars = {
        "TORCHINDUCTOR_WORKER_START": "fork",
        "QUANTIZATION_TYPE_DEBUG": "1",
    }
    executor = kubeflow_executor(
        namespace="test",
        nodes=2,
        num_gpus_per_node=4,
        custom_env_vars=recipe_env_vars,
    )

    assert executor.env_vars.items() >= recipe_env_vars.items()


@pytest.fixture
def mock_kubeflow_clients(monkeypatch):
    if not HAS_NEMO_RUN:
        pytest.skip("nemo_run not installed")
    from nemo_run.core.execution import kubeflow

    monkeypatch.setattr(kubeflow, "_KUBERNETES_AVAILABLE", True)
    monkeypatch.setattr(nemo_run.KubeflowExecutor, "_load_kube_clients", Mock())


@pytest.fixture
def diagnostic_executor(tmp_path, monkeypatch, mock_kubeflow_clients):
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path / "artifacts"))
    executor = DiagnosticKubeflowExecutor(
        launcher=nemo_run.Torchrun(nsys_profile=True),
        namespace="test",
        workdir_pvc="test-workdir",
        workdir_pvc_path="/workspace",
        num_nodes=16,
        gpus_per_node=4,
    )
    executor.assign("attempt-1", str(tmp_path / "run"), "training", "training")
    monkeypatch.setattr(executor, "_start_data_mover_pod", Mock())
    monkeypatch.setattr(executor, "_delete_data_mover_pod", Mock())
    monkeypatch.setattr(executor, "_rsync_from_pod", Mock())
    return executor


@pytest.mark.unit
def test_diagnostic_prefix_uses_one_pvc_output_and_unique_node_names(diagnostic_executor):
    executor = diagnostic_executor
    prefix = executor.get_launcher_prefix()
    assert prefix.count("-o") == 1
    assert prefix[prefix.index("-o") + 1] == (
        f"{executor.code_dir}/nsys_profile/profile_node%q{{PET_NODE_RANK}}_pid%p"
    )
    assert Path(executor.job_dir, "nsys_profile").is_dir()
    assert "--capture-range=cudaProfilerApi" in prefix
    assert executor.workdir_local_path is None


@pytest.mark.unit
def test_diagnostic_disabled_preserves_launcher_and_does_not_collect(diagnostic_executor):
    executor = diagnostic_executor
    executor.launcher.nsys_profile = False
    assert executor.get_launcher_prefix() is None
    assert executor.launcher.nsys_filename == "profile_%p"
    executor.cleanup("test-handle")
    executor._start_data_mover_pod.assert_not_called()
    assert not Path(executor.job_dir, "nsys_profile").exists()


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["unassigned", "no-pvc", "absolute-folder", "extra-output"])
def test_diagnostic_prefix_rejects_unrecoverable_configuration(diagnostic_executor, invalid):
    executor = diagnostic_executor
    if invalid == "unassigned":
        executor.job_dir = ""
    elif invalid == "no-pvc":
        executor.workdir_pvc = None
    elif invalid == "absolute-folder":
        executor.launcher.nsys_folder = "/tmp/profiles"
    else:
        executor.launcher.nsys_extra_args.append("--output=/tmp/other")
    with pytest.raises(ValueError):
        executor.get_launcher_prefix()


@pytest.mark.unit
def test_diagnostic_executor_survives_serialization_and_experiment_assignment(
    tmp_path, monkeypatch, mock_kubeflow_clients
):
    import fiddle as fdl
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    from nemo_run.run.torchx_backend.schedulers.api import REVERSE_EXECUTOR_MAPPING, get_executor_str

    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path / "nemo-run"))
    executor = DiagnosticKubeflowExecutor(
        launcher=nemo_run.Torchrun(nsys_profile=True), workdir_pvc="test", workdir_pvc_path="/workspace"
    )
    serializer = ZlibJSONSerializer()
    cloned = fdl.build(serializer.deserialize(serializer.serialize(executor.clone().to_config())))
    assert type(cloned) is DiagnosticKubeflowExecutor
    assert get_executor_str(cloned) == "kubeflow"
    assert REVERSE_EXECUTOR_MAPPING["kubeflow"] is nemo_run.KubeflowExecutor
    with nemo_run.Experiment("diagnostic-serialization", executor=cloned) as experiment:
        experiment.add(nemo_run.Script(path="/opt/Megatron-Bridge/scripts/performance/bootstrap.py"), name="train")
        assigned = experiment.jobs[0].executor
        assert type(assigned) is DiagnosticKubeflowExecutor
        prefix = assigned.get_launcher_prefix()
        assert assigned.experiment_id
        assert assigned.job_name == "train"
        assert prefix[prefix.index("-o") + 1].startswith(assigned.code_dir + "/nsys_profile/")


@pytest.mark.unit
def test_terminal_job_cleanup_preserves_collection_failure(diagnostic_executor, monkeypatch):
    from nemo_run.run.job import Job
    from torchx.specs import AppState

    executor = diagnostic_executor
    monkeypatch.setattr(executor, "_profile_inventory", Mock(side_effect=RuntimeError("inventory failed")))
    job = Job(id="train", task=nemo_run.Script(path="train.py"), executor=executor)
    job.handle = "kubeflow://test/diagnostic"
    job.state = AppState.RUNNING
    job.cleanup()
    executor._start_data_mover_pod.assert_not_called()
    job.state = AppState.SUCCEEDED
    job.cleanup()  # The framework suppresses cleanup errors; the manifest must survive.
    result = json.loads((executor._profile_destination() / "collection.json").read_text())
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert job.state is AppState.SUCCEEDED
    executor._delete_data_mover_pod.assert_called_once()


@pytest.mark.unit
def test_diagnostic_inventory_is_quoted_and_validated(diagnostic_executor, monkeypatch):
    invoke = Mock(return_value=SimpleNamespace(stdout="12\tprofile_node0_pid9.nsys-rep\n"))
    monkeypatch.setattr(executors_module.subprocess, "run", invoke)
    remote = "/workspace/path with spaces/nsys_profile"
    assert diagnostic_executor._profile_inventory("mover", remote) == [
        {"name": "profile_node0_pid9.nsys-rep", "bytes": 12}
    ]
    command = invoke.call_args.args[0]
    assert command[-1] == remote
    assert remote not in command[-3]
    assert invoke.call_args.kwargs["timeout"] == 120


@pytest.mark.unit
@pytest.mark.parametrize(
    "inventory",
    [
        "2\t../escape.nsys-rep\n",
        "-1\tprofile_node0_pid9.nsys-rep\n",
        "2\tprofile_node0_pid9.nsys-rep\n2\tprofile_node0_pid9.nsys-rep\n",
    ],
)
def test_diagnostic_rejects_unsafe_inventory(diagnostic_executor, monkeypatch, inventory):
    monkeypatch.setattr(executors_module.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=inventory)))
    with pytest.raises(ValueError):
        diagnostic_executor._profile_inventory("mover", "/workspace/profiles")


@pytest.mark.unit
def test_diagnostic_collects_attempts_separately_and_retains_empty_attempt(diagnostic_executor, monkeypatch):
    executor = diagnostic_executor
    inventory = Mock(return_value=[])
    monkeypatch.setattr(executor, "_profile_inventory", inventory)
    executor.cleanup("first")
    first = executor._profile_destination()
    assert json.loads((first / "collection.json").read_text())["status"] == "empty"

    executor.assign("attempt-2", executor.experiment_dir, "training", "second")
    inventory.return_value = [{"name": "profile_node15_pid7.nsys-rep", "bytes": 12}]

    def copy_report(pod, remote, destination):
        Path(destination, Path(remote).name).write_bytes(b"x" * 12)

    executor._rsync_from_pod.side_effect = copy_report
    executor.cleanup("second")
    second = executor._profile_destination()
    result = json.loads((second / "collection.json").read_text())
    assert second != first
    assert result["status"] == "collected"
    assert result["copied"] == ["profile_node15_pid7.nsys-rep"]
    assert (second / result["copied"][0]).stat().st_mode & 0o777 == 0o600
    assert json.loads((first / "collection.json").read_text())["status"] == "empty"
    assert executor._delete_data_mover_pod.call_count == 2
    assert executor._rsync_from_pod.call_args.args[1].endswith("/nsys_profile/profile_node15_pid7.nsys-rep")


@pytest.mark.unit
@pytest.mark.parametrize("condition", ["over_budget", "insufficient_space"])
def test_diagnostic_records_transfer_limits_without_copying(diagnostic_executor, monkeypatch, condition):
    executor = diagnostic_executor
    monkeypatch.setattr(
        executor, "_profile_inventory", Mock(return_value=[{"name": "profile_node0_pid7.nsys-rep", "bytes": 12}])
    )
    if condition == "over_budget":
        monkeypatch.setattr(executors_module, "_NSYS_REPORT_BUDGET_BYTES", 10)
    else:
        monkeypatch.setattr(executors_module.shutil, "disk_usage", Mock(return_value=SimpleNamespace(free=1)))
    executor.cleanup("test")
    result = json.loads((executor._profile_destination() / "collection.json").read_text())
    assert result["status"] == condition
    executor._rsync_from_pod.assert_not_called()
    executor._delete_data_mover_pod.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["start", "inventory", "copy", "mismatch", "delete"])
def test_diagnostic_records_failures_and_always_cleans_mover(diagnostic_executor, monkeypatch, failure):
    executor = diagnostic_executor
    monkeypatch.setattr(
        executor, "_profile_inventory", Mock(return_value=[{"name": "profile_node0_pid7.nsys-rep", "bytes": 12}])
    )
    error = RuntimeError("simulated failure")
    if failure == "start":
        executor._start_data_mover_pod.side_effect = error
    elif failure == "inventory":
        executor._profile_inventory.side_effect = subprocess.CalledProcessError(1, ["kubectl", "exec"])
    elif failure == "copy":
        executor._rsync_from_pod.side_effect = error
    else:
        size = 11 if failure == "mismatch" else 12
        executor._rsync_from_pod.side_effect = lambda pod, remote, destination: Path(
            destination, Path(remote).name
        ).write_bytes(b"x" * size)
        if failure == "delete":
            executor._delete_data_mover_pod.side_effect = error
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        executor.cleanup("test")
    result = json.loads((executor._profile_destination() / "collection.json").read_text())
    assert result["status"] == "failed"
    assert "error_type" in result
    executor._delete_data_mover_pod.assert_called_once()
