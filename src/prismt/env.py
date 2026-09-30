"""The computing environment: device choice, seeding, provenance and ``prismt doctor``.

``doctor`` is what the MATLAB Setup tab runs to decide whether a Python installation can
be used. It must never crash: PyTorch is first imported in a child process, so a broken
installation is reported as a failed check instead of killing the report.
"""

from __future__ import annotations

import functools

import importlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from prismt import FORMATS, __version__

REQUIRED = ("numpy", "scipy", "h5py", "sklearn", "torch")
OPTIONAL = ("optuna",)

_TORCH_PROBE = r"""
import json, sys
out = {"version": None, "cuda_build": None, "cuda_available": False, "cuda_devices": [],
       "mps_built": False, "mps_available": False}
import torch
out["version"] = torch.__version__
out["cuda_build"] = torch.version.cuda
out["cuda_available"] = bool(torch.cuda.is_available())
if out["cuda_available"]:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["cuda_devices"].append({"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1),
                                    "capability": f"{p.major}.{p.minor}"})
mps = getattr(torch.backends, "mps", None)
out["mps_built"] = bool(mps and mps.is_built())
out["mps_available"] = bool(mps and mps.is_available())
print(json.dumps(out))
"""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def select_device(requested: str = "auto") -> Any:
    """``auto`` picks CUDA, then Apple GPU (MPS) if it passes a quick self-test, then CPU."""
    import torch

    requested = (requested or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            from prismt.errors import EnvironmentProblem

            raise EnvironmentProblem(
                "E_ENV_NO_CUDA",
                "An NVIDIA GPU was requested but PyTorch cannot use one.",
                hint="Run on a GPU node, or install the CUDA build of PyTorch (see docs/cluster.md).",
            )
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_works():
            from prismt.errors import EnvironmentProblem

            raise EnvironmentProblem(
                "E_ENV_NO_MPS",
                "The Apple GPU (MPS) was requested but is not usable in this Python.",
                hint="Use an arm64 (not Rosetta) Python with PyTorch 2.2 or newer, or choose CPU.",
            )
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_works():
        return torch.device("mps")
    return torch.device("cpu")


_MPS_PROBE = """
import torch
a = torch.arange(12.0).reshape(3, 4)
assert torch.allclose(a @ a.T, (a.to("mps") @ a.to("mps").T).cpu())
torch.manual_seed(0)
lin = torch.nn.Linear(16, 16).to("mps")
x = torch.randn(2, 4, 8, 16, device="mps", requires_grad=True)
q = lin(x)
torch.nn.functional.scaled_dot_product_attention(q, q, q).sum().backward()
assert torch.isfinite(x.grad).all().item()
print("ok")
"""


@functools.lru_cache(maxsize=1)
def _mps_works() -> bool:
    """True when the Apple GPU can run what training needs (a layer, attention, a backward
    pass). Some virtual Macs report MPS as available but fail or even crash on it, so the
    probe runs in a child process. PRISMT_MPS=0 or 1 skips the probe."""
    import torch

    mps = getattr(torch.backends, "mps", None)
    if not (mps and mps.is_available()):
        return False
    forced = os.environ.get("PRISMT_MPS")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        out = subprocess.run([sys.executable, "-c", _MPS_PROBE], capture_output=True, text=True, timeout=180)
        return out.returncode == 0 and out.stdout.strip().endswith("ok")
    except Exception:  # noqa: BLE001 - any failure means "do not use MPS"
        return False


def git_info(path: Path | None = None) -> dict:
    """Commit and dirty flag of the repository holding the package (None outside git)."""
    root = Path(path or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=5).stdout.strip()) if commit else None
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def environment_record(device: Any = None) -> dict:
    """What run_info.json records about the software and hardware."""
    record = {
        "prismt": __version__,
        "formats": FORMATS,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "git": git_info(),
        "packages": {name: _version(name) for name in (*REQUIRED, *OPTIONAL)},
    }
    if device is not None:
        record["device"] = str(device)
        try:
            import torch

            if getattr(device, "type", "") == "cuda":
                record["gpu"] = torch.cuda.get_device_name(device)
        except Exception:  # noqa: BLE001
            pass
    return record


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a broken package counts as missing
        return None
    return str(getattr(mod, "__version__", "unknown"))


def _rosetta() -> bool:
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def doctor(*, require: str = "auto", out_dir: str | None = None, timeout: float = 180.0) -> dict:
    """Check that this Python can run PRISMT. Returns a JSON-able report; never raises."""
    checks: list[dict] = []
    warnings: list[dict] = []

    def check(name: str, ok: bool, message: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "message": message, "hint": hint})

    py_ok = sys.version_info >= (3, 10)
    check("python", py_ok, f"Python {sys.version.split()[0]} ({sys.executable})",
          "" if py_ok else "PRISMT needs Python 3.10 or newer; create the environment from environment.yml.")
    if _rosetta():
        warnings.append({"code": "W_ENV_ROSETTA", "message": "This Python runs under Rosetta (x86) on an "
                         "Apple silicon Mac, so the Apple GPU cannot be used.",
                         "hint": "Install the arm64 Miniforge and recreate the environment."})
    packages = {}
    for name in REQUIRED[:-1] + OPTIONAL:
        packages[name] = _version(name)
        if name in REQUIRED:
            check(name, packages[name] is not None, f"{name} {packages[name] or 'not installed'}",
                  "" if packages[name] else "Recreate the PRISMT environment (Setup tab → Create environment).")
    if packages.get("optuna") is None:
        warnings.append({"code": "W_ENV_NO_OPTUNA", "message": "optuna is not installed, so automatic "
                         "tuning (HPO) is unavailable.", "hint": "pip install optuna"})

    torch_info: dict[str, Any] = {}
    try:
        proc = subprocess.run([sys.executable, "-c", _TORCH_PROBE], capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            torch_info = json.loads(proc.stdout.strip().splitlines()[-1])
            check("torch", True, f"PyTorch {torch_info['version']}")
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            check("torch", False, "PyTorch could not be imported: " + " | ".join(tail),
                  "Recreate the PRISMT environment (Setup tab → Create environment).")
    except subprocess.TimeoutExpired:
        check("torch", False, f"Importing PyTorch took longer than {timeout:.0f} s.",
              "Try again; if it keeps happening, recreate the environment.")
    packages["torch"] = torch_info.get("version")

    device = "cpu"
    if torch_info:
        if torch_info.get("cuda_available"):
            device = "cuda"
        elif torch_info.get("mps_available"):
            try:
                device = "mps" if _mps_works() else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
            if device == "cpu":
                warnings.append({"code": "W_ENV_MPS_FAILED", "message": "The Apple GPU is present but failed "
                                 "a self-test; PRISMT will use the CPU.", "hint": ""})
        req = (require or "auto").lower()
        if req in ("cuda", "mps") and device != req:
            check("device", False, f"A {req.upper()} device was required but PRISMT would use {device}.",
                  "Run on a machine with that device, or choose 'auto'.")
        else:
            check("device", True, {"cuda": "NVIDIA GPU", "mps": "Apple GPU (MPS)", "cpu": "CPU"}[device])

    if out_dir:
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=out_dir):
                pass
            check("output_folder", True, f"Can write to {out_dir}")
        except OSError as exc:
            check("output_folder", False, f"Cannot write to {out_dir} ({exc})", "Choose another folder.")

    return {
        "ok": all(c["ok"] for c in checks),
        "prismt": {"version": __version__, "path": str(Path(__file__).resolve().parent), "formats": FORMATS},
        "python": {"version": sys.version.split()[0], "executable": sys.executable,
                   "platform": platform.platform(), "machine": platform.machine()},
        "packages": packages,
        "torch": torch_info,
        "device": device,
        "checks": checks,
        "warnings": warnings,
        "environment": {"PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                        "launched_by": os.environ.get("PRISMT_LAUNCHED_BY", "")},
    }
