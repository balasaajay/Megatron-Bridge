"""CPU validation for the isolated, image-preserving cyclic-GC diagnostic."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


PERFORMANCE_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(PERFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(PERFORMANCE_DIR))

import setup_experiment
from argument_parser import parse_cli_args
from utils import gc_timing_recorder as recorder_module
from utils.gc_timing_recorder import EVENT_CAPACITY, GCTimingCallback
from utils.gc_timing_stage import (
    BOOTSTRAP_IMPORT,
    BOOTSTRAP_PRELUDE,
    FROZEN_ENTRYPOINT_HASHES,
    INSTRUMENTED_PRETRAIN_CALL,
    PRETRAIN_CALL,
    GCTimingKubeflowExecutor,
    stage_gc_timing,
)

from megatron.bridge.training.callbacks import CallbackContext, normalize_callbacks
from megatron.bridge.training.utils.log_utils import setup_logging


pytestmark = pytest.mark.unit


@pytest.fixture
def context():
    return CallbackContext(state=SimpleNamespace(train_state=SimpleNamespace(step=0)), model=[])


@pytest.fixture
def probes(monkeypatch, caplog):
    created = []
    monkeypatch.setattr(recorder_module, "get_rank_safe", lambda: 7)
    caplog.set_level(logging.INFO, logger=recorder_module.logger.name)

    def make(*, capacity=EVENT_CAPACITY, long_event_ns=0):
        probe = GCTimingCallback(capacity=capacity, long_event_ns=long_event_ns)
        created.append(probe)
        return probe

    yield make
    for probe in created:
        probe.detach()


def reports(caplog):
    return [
        json.loads(record.getMessage().split("GC_TIMING ", 1)[1])
        for record in caplog.records
        if "GC_TIMING " in record.getMessage()
    ]


def test_real_gc_lifecycle_and_body_interstep_scope(probes, context, caplog):
    before = list(gc.callbacks)
    policy = (gc.isenabled(), gc.get_threshold())
    probe = probes()
    manager = normalize_callbacks([probe])
    manager.fire("on_train_start", context)
    assert len(probe.events) * probe.events.itemsize == 655360
    assert len(gc.callbacks) == len(before) + 1
    for step in range(2):
        context.state.train_state.step = step
        manager.fire("on_train_step_start", context)
        cycle = []
        cycle.append(cycle)
        del cycle
        gc.collect(0)
        manager.fire("on_train_step_end", context)
        context.state.train_state.step += 1
        gc.collect(0)
    manager.fire("on_train_end", context)
    assert gc.callbacks == before
    assert reports(caplog) == []  # Nothing logs during measurement or detach.
    probe.close()
    probe.close()
    records = reports(caplog)
    summary = records[0]
    assert summary["kind"] == "summary"
    assert summary["rank"] == 7
    assert summary["collections"] >= 4
    assert summary["capture_complete"] == 1
    assert summary["existing_callbacks"] == len(before)
    assert summary["attachment_thread"] == threading.get_native_id()
    bodies = [record for record in records if record["kind"] == "interval" and record["phase"] == 1]
    assert [body["iteration"] for body in bodies] == [1, 2]
    assert [body["start_counter"] for body in bodies] == [0, 1]
    assert [body["stop_counter"] for body in bodies] == [0, 1]
    assert all(body["wall_ns"] == body["stop_ns"] - body["start_ns"] for body in bodies)
    assert any(record.get("start_phase") == 2 for record in records if record["kind"] == "gc")
    assert (gc.isenabled(), gc.get_threshold()) == policy
    assert sum(record["kind"] == "summary" for record in records) == 1


def test_identity_detach_preserves_foreign_callback(probes, context):
    foreign = lambda phase, info: None
    gc.callbacks.append(foreign)
    try:
        probe = probes()
        probe.on_train_start(context)
        probe.detach()
        probe.detach()
        assert any(callback is foreign for callback in gc.callbacks)
        assert all(callback is not probe._callback for callback in gc.callbacks)
    finally:
        gc.callbacks.remove(foreign)


def test_exception_and_missing_capture_are_explicit(probes, context, caplog):
    probe = probes()
    try:
        probe.on_train_start(context)
        probe.on_train_step_start(context)
        raise RuntimeError("training failure")
    except RuntimeError:
        pass
    finally:
        probe.close()
    summary = reports(caplog)[0]
    assert summary["capture_complete"] == 0
    assert summary["train_end_seen"] == 0
    assert summary["unpaired_body_endpoints"] == 1
    assert probe._callback not in gc.callbacks
    caplog.clear()
    probes().close()
    assert reports(caplog)[0]["started"] == 0


def test_fixed_buffer_overflow_retains_explicit_missing_pair(probes, context, caplog):
    probe = probes(capacity=2)
    original_size = len(probe.events)
    probe.on_train_start(context)
    probe._on_gc("start", {"generation": 2})
    probe._on_gc("stop", {"generation": 2, "collected": 3, "uncollectable": 0})
    probe.on_train_end(context)
    probe.close()
    summary = reports(caplog)[0]
    assert len(probe.events) == original_size
    assert summary["events"] == 2
    assert summary["dropped_events"] == 2
    assert summary["unpaired_gc_endpoints"] == 1
    assert summary["capture_complete"] == 0


@pytest.mark.parametrize("capacity", [0, 1, 13108])
def test_rejects_invalid_numeric_budget(capacity):
    with pytest.raises(ValueError):
        GCTimingCallback(capacity=capacity, long_event_ns=0)


def test_background_collector_thread_is_preserved(probes, context, caplog):
    probe = probes()
    probe.on_train_start(context)
    worker = threading.Thread(target=gc.collect, args=(0,))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    probe.on_train_end(context)
    probe.close()
    records = reports(caplog)
    assert records[0]["attachment_thread"] == records[0]["main_thread"]
    assert any(record.get("thread") == worker.native_id for record in records if record["kind"] == "gc")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="PID inheritance requires fork")
def test_real_fork_does_not_record_or_emit_parent_evidence(probes, context):
    probe = probes()
    probe.on_train_start(context)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        before = probe.count
        gc.collect(0)
        probe.close()
        os.write(write_fd, bytes([int(probe.count == before), int(not probe.reported)]))
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 2) == b"\x01\x01"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        os.close(read_fd)


def test_largest_late_gc_events_survive_output_bound(probes, context, monkeypatch, caplog):
    probe = probes()
    clock = [10_000]
    monkeypatch.setattr(probe, "_wall_clock", lambda: clock[0])
    monkeypatch.setattr(probe, "_cpu_clock", lambda: clock[0])
    monkeypatch.setattr(recorder_module, "MAX_OUTPUT_LONG_EVENTS", 2)
    probe.on_train_start(context)
    for duration in [2, 3, 100, 200]:
        clock[0] += 1000
        probe._on_gc("start", {"generation": 2})
        clock[0] += duration
        probe._on_gc("stop", {"generation": 2, "collected": 1, "uncollectable": 0})
    probe.on_train_end(context)
    probe.close()
    records = reports(caplog)
    assert records[0]["capture_complete"] == 1
    assert records[0]["output_complete"] == 0
    assert records[0]["omitted_long_events"] == 2
    assert [record["stop_ns"] - record["start_ns"] for record in records if record["kind"] == "gc"] == [100, 200]


def test_actual_bridge_logging_keeps_all_rank_summaries(probes, context, caplog, monkeypatch):
    loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    original = [(log, log.level, list(log.filters)) for log in loggers]
    try:
        setup_logging(logging_level=logging.INFO, filter_warning=True)
        for rank in range(64):
            monkeypatch.setattr(recorder_module, "get_rank_safe", lambda rank=rank: rank)
            probe = probes(capacity=8)
            probe.on_train_start(context)
            probe.on_train_end(context)
            probe.close()
        assert {record["rank"] for record in reports(caplog) if record["kind"] == "summary"} == set(range(64))
    finally:
        for log, level, filters in original:
            log.setLevel(level)
            log.filters[:] = filters


def test_cpu_observer_overhead_is_measured(probes, context, caplog):
    """A bounded CPU fixture measures observer cost, not distributed overhead."""
    samples = {False: [], True: []}
    collections_per_sample = 100
    for repeat in range(4):
        for instrumented in [False, True] if repeat % 2 == 0 else [True, False]:
            probe = probes() if instrumented else None
            if probe:
                probe.on_train_start(context)
            started = time.perf_counter_ns()
            for _ in range(collections_per_sample):
                values = [[] for _ in range(100)]
                for value in values:
                    value.append(value)
                del values, value
                gc.collect(0)
            elapsed = time.perf_counter_ns() - started
            if probe:
                probe.on_train_end(context)
                probe.close()
            samples[instrumented].append(elapsed)
    baseline = statistics.median(samples[False])
    observed = statistics.median(samples[True])
    overhead = (observed - baseline) / collections_per_sample
    logging.getLogger(__name__).info(
        "GC_TIMING_CPU_OVERHEAD %s",
        json.dumps(
            {
                "baseline_ns": baseline,
                "observed_ns": observed,
                "collections": collections_per_sample,
                "delta_ns_per_forced_collection": overhead,
                "baseline_samples_ns": samples[False],
                "observed_samples_ns": samples[True],
            }
        ),
    )
    # Broad reliability ceiling only; measured overhead is reported for review.
    assert overhead < 1_000_000


def test_stage_is_exact_and_fails_closed(tmp_path):
    stage = stage_gc_timing(destination=tmp_path / "stage", performance_dir=PERFORMANCE_DIR)
    assert sorted(path.name for path in (stage / "gc_probe").iterdir()) == [
        "bootstrap.py",
        "gc_timing_recorder.py",
        "run_recipe.py",
    ]
    assert (stage / "gc_probe/bootstrap.py").read_text() == (PERFORMANCE_DIR / "bootstrap.py").read_text().replace(
        BOOTSTRAP_IMPORT, BOOTSTRAP_PRELUDE
    )
    original = (PERFORMANCE_DIR / "run_recipe.py").read_text()
    assert (stage / "gc_probe/run_recipe.py").read_text() == original.replace(
        PRETRAIN_CALL, INSTRUMENTED_PRETRAIN_CALL
    )
    assert stage_gc_timing(destination=stage, performance_dir=PERFORMANCE_DIR) == stage
    (stage / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="only the exact"):
        stage_gc_timing(destination=stage, performance_dir=PERFORMANCE_DIR)
    fake_source = tmp_path / "source"
    fake_source.mkdir()
    (fake_source / "bootstrap.py").write_text("wrong source")
    with pytest.raises(ValueError, match="exact frozen"):
        stage_gc_timing(destination=tmp_path / "bad", performance_dir=fake_source)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("inherited", [None, "", "/image/mcore", "/image/mcore:/image/extra"])
def test_bootstrap_prelude_preserves_inherited_paths_without_adding_cwd(inherited):
    environment = {} if inherited is None else {"PYTHONPATH": inherited}
    paths = ["/staged", "/site-packages"]
    namespace = {
        "sys": SimpleNamespace(path=paths),
        "os": SimpleNamespace(environ=environment, pathsep=os.pathsep),
    }
    exec(BOOTSTRAP_PRELUDE.split(BOOTSTRAP_IMPORT)[0], namespace)
    assert paths == [
        "/staged",
        "/opt/Megatron-Bridge/scripts/performance",
        "/opt/Megatron-Bridge/src",
        "/site-packages",
    ]
    expected = ["/opt/Megatron-Bridge/scripts/performance", "/opt/Megatron-Bridge/src"]
    if inherited:
        expected.extend(inherited.split(os.pathsep))
    assert environment["PYTHONPATH"].split(os.pathsep) == expected


@pytest.mark.parametrize("option", ["use_recipes", "enable_nsys", "pytorch_profiler", "record_memory_history"])
def test_launcher_rejects_incompatible_modes_before_staging(tmp_path, option):
    arguments = {name: None for name in inspect.signature(setup_experiment.main).parameters}
    arguments.update(
        model_family_name="nemotronh",
        kubeflow_namespace="test",
        kubeflow_gc_timing_stage=str(tmp_path / "stage"),
        use_recipes=True,
        task="pretrain",
    )
    arguments[option] = False if option == "use_recipes" else True
    with pytest.raises(ValueError, match="frozen recipe pretrain entrypoint"):
        setup_experiment.main(**arguments)
    assert not (tmp_path / "stage").exists()


def test_launcher_filter_and_script_keep_image_pythonpath(tmp_path):
    stage = tmp_path / "stage"
    args = [
        "--kubeflow_gc_timing_stage",
        str(stage),
        "--model_family_name",
        "nemotronh",
        "--kubeflow_workdir_local_path=" + str(stage),
        "--model_recipe_name",
        "nemotron_nano_12b_v2",
        "--num_gpus",
        "64",
        "--gpu",
        "gb200",
    ]
    filtered = setup_experiment._filter_run_script_args(args)
    assert filtered == [
        "--model_family_name",
        "nemotronh",
        "--model_recipe_name",
        "nemotron_nano_12b_v2",
        "--num_gpus",
        "64",
        "--gpu",
        "gb200",
    ]
    parsed, unknown = parse_cli_args().parse_known_args(args)
    assert parsed.kubeflow_gc_timing_stage == str(stage)
    assert unknown == []
    script = setup_experiment._build_nemorun_script(
        script_path="/nemo_run/gc_probe/bootstrap.py",
        script_dir="/opt/Megatron-Bridge/scripts/performance",
        args=filtered,
        kubeflow_namespace="test",
        custom_env_vars={},
    )
    assert script.path == "/nemo_run/gc_probe/bootstrap.py"
    assert script.env["PYTHONPATH"] == "/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:$PYTHONPATH"
    assert script.args == filtered


@pytest.fixture
def frozen_kubeflow_clients(monkeypatch):
    import nemo_run as run
    from nemo_run.core.execution import kubeflow as kubeflow_module

    monkeypatch.setattr(kubeflow_module, "_KUBERNETES_AVAILABLE", True)
    monkeypatch.setattr(run.KubeflowExecutor, "_load_kube_clients", lambda self: None)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_actual_experiment_run_packages_without_rsync(tmp_path, monkeypatch, frozen_kubeflow_clients, diagnostic):
    import fiddle as fdl
    import nemo_run as run
    import nemo_run.config as run_config
    from nemo_run.core.execution.kubeflow import KubeflowJobState
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    from nemo_run.run.torchx_backend.schedulers.api import REVERSE_EXECUTOR_MAPPING, get_executor_str
    from torchx.specs import AppState

    assert shutil.which("rsync") is None, "Run this proof in the rsync-free tooling container."
    monkeypatch.setattr(run_config, "_NEMORUN_HOME", str(tmp_path / "nemorun"))
    stage = tmp_path / "stage"
    executor_cls = GCTimingKubeflowExecutor if diagnostic else run.KubeflowExecutor
    executor = setup_experiment.kubeflow_executor(
        namespace="test",
        nodes=16,
        num_gpus_per_node=4,
        container_image="example.invalid/frozen:test",
        workdir_pvc="test",
        executor_cls=executor_cls,
    )
    if diagnostic:
        executor.gc_staging_dir = str(stage)
        executor.gc_performance_dir = str(PERFORMANCE_DIR)
    serializer = ZlibJSONSerializer()
    executor = fdl.build(serializer.deserialize(serializer.serialize(executor.clone().to_config())))
    assert type(executor) is executor_cls
    if diagnostic:
        assert executor.gc_staging_dir == str(stage)
        assert executor.gc_performance_dir == str(PERFORMANCE_DIR)
    assert get_executor_str(executor) == "kubeflow"
    assert REVERSE_EXECUTOR_MAPPING["kubeflow"] is run.KubeflowExecutor
    movers = []
    uploaded = []
    monkeypatch.setattr(run.KubeflowExecutor, "_start_data_mover_pod", lambda self, pod: movers.append(("start", pod)))
    monkeypatch.setattr(
        run.KubeflowExecutor, "_delete_data_mover_pod", lambda self, pod: movers.append(("delete", pod))
    )

    def upload(self, pod, local, remote):
        directory = Path(local)
        assert (directory / "launch.sh").is_file()
        assert (directory / "gc_probe").exists() is diagnostic
        assert not (directory / "src").exists() and not (directory / "3rdparty").exists()
        if diagnostic:
            assert sorted(path.name for path in (directory / "gc_probe").iterdir()) == [
                "bootstrap.py",
                "gc_timing_recorder.py",
                "run_recipe.py",
            ]
            for path in (stage / "gc_probe").iterdir():
                assert (directory / "gc_probe" / path.name).read_bytes() == path.read_bytes()
        uploaded.append((local, remote))

    monkeypatch.setattr(run.KubeflowExecutor, "_rsync_to_pod", upload)
    launch = Mock(return_value=("gc-proof-job", KubeflowJobState.CREATED))
    monkeypatch.setattr(run.KubeflowExecutor, "launch", launch)
    monkeypatch.setattr(run.KubeflowExecutor, "status", Mock(return_value=KubeflowJobState.SUCCEEDED))
    task = setup_experiment._build_nemorun_script(
        script_path="/nemo_run/gc_probe/bootstrap.py"
        if diagnostic
        else "/opt/Megatron-Bridge/scripts/performance/bootstrap.py",
        script_dir="/opt/Megatron-Bridge/scripts/performance",
        args=["--use_recipes"],
        kubeflow_namespace="test",
        custom_env_vars={},
    )
    with run.Experiment("gc-proof", executor=executor) as experiment:
        experiment.add(task, name="train")
        job = experiment.jobs[0]
        assert not Path(job.executor.job_dir).exists()  # No setup-time parent creation.
        experiment.run()
        assert job.launched and job.handle.startswith("kubeflow://")
        assert experiment.detach is False
    assert job.state is AppState.SUCCEEDED
    assert len(uploaded) == 1
    assert movers == [("start", movers[0][1]), ("delete", movers[0][1])]
    assert job.executor.workdir_local_path is None
    launch.assert_called_once_with(name="train", cmd=["/bin/bash", f"{job.executor.code_dir}/launch.sh"])
    launch_text = Path(job.executor.job_dir, "launch.sh").read_text()
    assert task.path in launch_text
    assert "PYTHONPATH" not in job.executor.env_vars
    assert "export PYTHONPATH" not in launch_text
    assert not (Path(job.executor.job_dir) / "gc_probe" / "utils").exists()
    if diagnostic:
        # Native copy_to_workspace deletes its mover even when upload fails.
        def fail_upload(self, pod, local, remote):
            raise RuntimeError("expected-upload-failure")

        monkeypatch.setattr(run.KubeflowExecutor, "_rsync_to_pod", fail_upload)
        with pytest.raises(RuntimeError, match="expected-upload-failure"):
            job.executor.package(job.executor.packager, "train")
        assert movers[-2:] == [("start", movers[-1][1]), ("delete", movers[-1][1])]
        (Path(job.executor.job_dir) / "gc_probe" / "run_recipe.py").write_text("wrong")
        with pytest.raises(ValueError, match="different GC timing bundle"):
            job.executor.package(job.executor.packager, "train")


def test_actual_launcher_and_staged_bootstrap_handoff(tmp_path, monkeypatch, frozen_kubeflow_clients):
    """Exercise the real image parser/utils, substituting only GPU training work."""
    stage = tmp_path / "stage"
    arguments = {name: None for name in inspect.signature(setup_experiment.main).parameters}
    arguments.update(
        use_recipes=True,
        model_family_name="nemotronh",
        model_recipe_name="nemotron_nano_12b_v2",
        task="pretrain",
        compute_dtype="bf16",
        gpu="gb200",
        num_gpus=64,
        gpus_per_node=4,
        custom_mounts=[],
        custom_env_vars={},
        kubeflow_namespace="test",
        kubeflow_workdir_pvc="test",
        kubeflow_workdir_pvc_path="/nemo_run",
        container_image="example.invalid/frozen:test",
        max_retries=0,
        kubeflow_gc_timing_stage=str(stage),
    )
    cli = [
        "--use_recipes",
        "--model_family_name",
        "nemotronh",
        "--model_recipe_name",
        "nemotron_nano_12b_v2",
        "--num_gpus",
        "64",
        "--gpu",
        "gb200",
        "--kubeflow_gc_timing_stage",
        str(stage),
    ]
    monkeypatch.setattr(sys, "argv", ["setup_experiment.py", *cli])
    captured = {}
    real_executor_factory = setup_experiment.kubeflow_executor

    def capture_executor(**kwargs):
        captured["executor_kwargs"] = kwargs
        captured["executor"] = real_executor_factory(**kwargs)
        return captured["executor"]

    monkeypatch.setattr(setup_experiment, "kubeflow_executor", capture_executor)
    real_builder = setup_experiment._build_nemorun_script

    class StopBeforeSubmission(Exception):
        pass

    def capture_task(**kwargs):
        captured["task"] = real_builder(**kwargs)
        return captured["task"]

    monkeypatch.setattr(setup_experiment, "_build_nemorun_script", capture_task)
    monkeypatch.setattr(setup_experiment, "HAVE_WANDB", False)

    def stop_before_submission(*args, **kwargs):
        raise StopBeforeSubmission

    monkeypatch.setattr(setup_experiment.run, "run", stop_before_submission)
    with pytest.raises(StopBeforeSubmission):
        setup_experiment.main(**arguments)
    task = captured["task"]
    assert captured["executor"].workdir_local_path is None
    assert type(captured["executor"]) is GCTimingKubeflowExecutor
    assert captured["executor"].gc_staging_dir == str(stage)
    assert "PYTHONPATH" not in captured["executor"].env_vars
    assert task.path == "/nemo_run/gc_probe/bootstrap.py"
    assert not any(argument.startswith("--kubeflow_") for argument in task.args)
    assert task.env["PYTHONPATH"].startswith("/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:")

    image_performance = Path("/opt/Megatron-Bridge/scripts/performance")
    assert image_performance.is_dir(), "Run this proof in the container with the frozen image source mount."
    for name, expected in FROZEN_ENTRYPOINT_HASHES.items():
        assert hashlib.sha256((image_performance / name).read_bytes()).hexdigest() == expected
    harness = tmp_path / "entrypoint_proof.py"
    harness.write_text(
        textwrap.dedent("""
        import gc
        import json
        import logging
        import os
        import sys
        import types
        from pathlib import Path
        from types import SimpleNamespace

        logging.basicConfig(level=logging.INFO)
        stage = Path(sys.argv[1])
        forwarded = json.loads(sys.argv[2])
        assert os.environ["PYTHONPATH"] == "/mcore"
        sys.path.insert(0, str(stage / "gc_probe"))
        import bootstrap
        assert sys.path[0] == str(stage / "gc_probe")
        assert sys.path[1:3] == ["/opt/Megatron-Bridge/scripts/performance", "/opt/Megatron-Bridge/src"]
        assert os.environ["PYTHONPATH"] == "/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:/mcore"
        import argument_parser
        import run_recipe
        import utils.utils as image_utils
        assert argument_parser.__file__.startswith("/opt/Megatron-Bridge/scripts/performance/")
        assert image_utils.__file__.startswith("/opt/Megatron-Bridge/scripts/performance/")
        assert bootstrap.__file__ == str(stage / "gc_probe/bootstrap.py")
        assert run_recipe.__file__ == str(stage / "gc_probe/run_recipe.py")
        assert "megatron.bridge.training.pretrain" not in sys.modules
        results = []
        for mode in ("success", "exception", "manual_policy"):
            # Each case models a fresh bootstrap; imported training packages may
            # mutate the environment when exec is replaced by this CPU handoff.
            os.environ["PYTHONPATH"] = "/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:/mcore"
            seen = []
            recipe = SimpleNamespace(env_vars={"GC_BOOTSTRAP_SENTINEL": "ready"},
                model=SimpleNamespace(moe_flex_dispatcher_backend=None),
                optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False),
                train=SimpleNamespace(manual_gc=mode == "manual_policy"), print_yaml=lambda: None)
            preparation_phases = []
            def prepare_recipe(args, overrides, *, environment_only):
                assert args.use_recipes and args.task == "pretrain"
                preparation_phases.append(environment_only)
                return recipe
            run_recipe._prepare_recipe = prepare_recipe
            sys.argv = [str(stage / "gc_probe/bootstrap.py"), *forwarded]

            def handoff(executable, argv, env):
                assert env["GC_BOOTSTRAP_SENTINEL"] == "ready"
                assert argv[1] == str(stage / "gc_probe/run_recipe.py")
                from megatron.bridge.training.callbacks import CallbackContext, normalize_callbacks
                import megatron.bridge.training.callbacks as callbacks_module
                assert callbacks_module.__file__.startswith("/opt/Megatron-Bridge/src/")
                import megatron.core as mcore_module
                assert mcore_module.__file__.startswith("/mcore/")
                assert env["PYTHONPATH"] == "/opt/Megatron-Bridge/scripts/performance:/opt/Megatron-Bridge/src:/mcore"
                def pretrain(*, config, forward_step_func, callbacks):
                    assert config is recipe
                    assert len(callbacks) == 1
                    probe = callbacks[0]
                    assert not probe.active
                    seen.append(probe)
                    context = CallbackContext(state=SimpleNamespace(train_state=SimpleNamespace(step=0)), model=[])
                    manager = normalize_callbacks(callbacks)
                    manager.fire("on_train_start", context)
                    manager.fire("on_train_step_start", context)
                    gc.collect(0)
                    if mode == "exception":
                        raise RuntimeError("expected-training-error")
                    manager.fire("on_train_step_end", context)
                    context.state.train_state.step = 1
                    manager.fire("on_train_end", context)
                stubs = {
                    "megatron.bridge.training.gpt_step": {"forward_step": object},
                    "megatron.bridge.training.pretrain": {"pretrain": pretrain},
                }
                for name, attributes in stubs.items():
                    module = types.ModuleType(name)
                    module.__dict__.update(attributes)
                    sys.modules[name] = module
                sys.argv = argv[1:]
                run_recipe.main()
            bootstrap.os.execvpe = handoff
            caught = None
            try:
                bootstrap.main()
            except (RuntimeError, ValueError) as error:
                caught = str(error)
            assert preparation_phases == [True, False]
            if mode == "manual_policy":
                assert not seen
                assert caught == "GC timing requires the frozen automatic-GC workload policy."
            else:
                assert len(seen) == 1 and seen[0].reported and not seen[0].active
                assert all(callback is not seen[0]._callback for callback in gc.callbacks)
                assert caught == ("expected-training-error" if mode == "exception" else None)
            results.append({"mode": mode, "callbacks": len(seen), "cleanup": True})
        logging.getLogger(__name__).info("GC_ENTRYPOINT_PROOF %s", json.dumps(results))
    """)
    )
    environment = os.environ.copy()
    # Exact read-only MCore stands in for the image's editable installation.
    # The bootstrap must add image paths without inheriting the launcher checkout.
    environment["PYTHONPATH"] = "/mcore"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(harness), str(stage), json.dumps(task.args)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GC_ENTRYPOINT_PROOF" in result.stderr
    summaries = [
        json.loads(line.split("GC_TIMING ", 1)[1]) for line in result.stderr.splitlines() if "GC_TIMING " in line
    ]
    assert [record["capture_complete"] for record in summaries if record["kind"] == "summary"] == [1, 0]
