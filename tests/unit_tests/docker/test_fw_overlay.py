"""Check Framework overlay preservation without downloading GPU packages."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import tomllib


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("version", ["0.6.18.post1", "0.7.2", "0.7.2.post1"])
def test_fw_overlay_preserves_flashinfer_pair(tmp_path, version):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to exercise the Framework overlay")
    config = tomllib.loads((ROOT / "docker/common/fw_pyproject.toml").read_text())
    overrides = config["tool"]["uv"]["override-dependencies"]
    flashinfer_overrides = [item for item in overrides if item.startswith("flashinfer-")]
    dockerfile = (ROOT / "docker/Dockerfile.fw_final").read_text().replace("\\\n", " ")
    assert re.search(r"uv sync[^\n]*--inexact", dockerfile)

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / ".venv")
    env.pop("VIRTUAL_ENV", None)
    env["UV_CACHE_DIR"] = str(tmp_path / "cache")

    # uv resolves the synthetic 'never' platform too. Supply metadata-only wheels
    # so the real resolver can run offline, including with the old Python-only pin.
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for package in ("flashinfer_python", "flashinfer_cubin"):
        for candidate in ("0.6.18.post1", "0.7.2", "0.7.2.post1"):
            info = f"{package}-{candidate}.dist-info"
            with ZipFile(wheels / f"{package}-{candidate}-py3-none-any.whl", "w") as wheel:
                wheel.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {package}\nVersion: {candidate}\n")
                wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                wheel.writestr(
                    f"{info}/RECORD", "".join(f"{info}/{name},,\n" for name in ("METADATA", "WHEEL", "RECORD"))
                )

    def run(*args):
        result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return result.stdout

    run(uv, "venv", "--python", sys.executable, str(tmp_path / ".venv"))
    python = str(tmp_path / ".venv/bin/python")
    run(
        uv,
        "pip",
        "install",
        "--python",
        python,
        "--no-index",
        "--find-links",
        str(wheels),
        f"flashinfer-python=={version}",
        f"flashinfer-cubin=={version}",
    )

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "overlay-test"\nversion = "0.0.0"\nrequires-python = ">=3.12"\n'
        'dependencies = ["flashinfer-python"]\n[tool.uv]\noverride-dependencies = '
        + json.dumps(flashinfer_overrides)
        + "\n"
    )
    run(
        uv,
        "sync",
        "--inexact",
        "--offline",
        "--no-index",
        "--find-links",
        str(wheels),
        "--no-install-project",
        "--python",
        sys.executable,
    )
    installed = json.loads(
        run(
            python,
            "-c",
            "import json; from importlib.metadata import version; "
            "print(json.dumps([version('flashinfer-python'), version('flashinfer-cubin')]))",
        )
    )
    assert installed == [version, version]
