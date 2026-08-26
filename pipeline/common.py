"""Shared plumbing: paths, checkpointing, timing, peak-memory sampling.

Every number that ends up in benchmark.json / RESULTS.md is produced here by
actual measurement. Nothing in this file estimates anything.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
OUT = ROOT / "outputs"
TOOLS = ROOT / "tools"
LOGS = ROOT / "logs"


# --------------------------------------------------------------------------- paths
@dataclass
class Paths:
    ds: str

    @property
    def images(self) -> Path:          return DATA / self.ds / "images"
    @property
    def work(self) -> Path:            return WORK / self.ds
    @property
    def resized(self) -> Path:         return self.work / "images_resized"
    @property
    def frames(self) -> Path:          return self.work / "frames_jpg"
    @property
    def masks(self) -> Path:           return self.work / "masks"
    @property
    def masked_images(self) -> Path:   return self.work / "images_masked"
    @property
    def mvs(self) -> Path:             return self.work / "mvs"
    @property
    def state(self) -> Path:           return self.work / ".state"
    @property
    def out(self) -> Path:             return OUT / self.ds

    def colmap(self, variant: str) -> Path:
        return self.work / f"colmap_{variant}"

    def ensure(self) -> "Paths":
        for p in (self.work, self.state, self.out):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- host
def host_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "cpu_threads": os.cpu_count(),
        "ram_gb": None,
        "gpu": "none-detected",
        "cuda": False,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            name = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            if name:
                info["cpu"] = name.splitlines()[0].strip()
            gpu = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            if gpu:
                info["gpu"] = " / ".join(g.strip() for g in gpu.splitlines() if g.strip())
        except Exception:
            pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


def device() -> str:
    """cuda when genuinely present, else cpu. Every stage honours this."""
    if os.environ.get("MANAH_FORCE_CPU"):
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- sampler
class _MemSampler(threading.Thread):
    """Polls this process + all descendants, tracking peak combined RSS.

    Child processes matter here: colmap.exe / DensifyPointCloud.exe do the heavy
    allocation, not the python parent.
    """

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._halt = threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except Exception:
            return
        me = psutil.Process()
        while not self._halt.is_set():
            total = 0
            try:
                procs = [me] + me.children(recursive=True)
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except Exception:
                        pass
            except Exception:
                pass
            self.peak_rss_mb = max(self.peak_rss_mb, total / 1024**2)
            self._halt.wait(self.interval)

    def stop(self) -> float:
        self._halt.set()
        self.join(timeout=2)
        return round(self.peak_rss_mb, 1)


# --------------------------------------------------------------------------- stage
@dataclass
class StageResult:
    name: str
    seconds: float
    peak_vram_mb: float
    peak_ram_mb: float
    status: str = "ok"
    notes: dict = field(default_factory=dict)


class Stage:
    """Context manager: times a stage, samples peak RAM/VRAM, persists the record."""

    def __init__(self, paths: Paths, name: str):
        self.paths = paths
        self.name = name
        self.notes: dict = {}
        self.status = "ok"

    def __enter__(self) -> "Stage":
        self.t0 = time.perf_counter()
        self.sampler = _MemSampler()
        self.sampler.start()
        self._cuda = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._cuda = True
        except Exception:
            pass
        log(f"[{self.name}] start")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        seconds = round(time.perf_counter() - self.t0, 2)
        peak_ram = self.sampler.stop()
        vram = 0.0
        if self._cuda:
            try:
                import torch
                vram = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            except Exception:
                pass
        if exc_type is not None:
            self.status = "failed"
            self.notes["error"] = f"{exc_type.__name__}: {exc}"
        res = StageResult(self.name, seconds, vram, peak_ram, self.status, self.notes)
        write_stage(self.paths, res)
        log(f"[{self.name}] {res.status} in {seconds}s  peak_ram={peak_ram}MB peak_vram={vram}MB")
        return False  # never swallow


def write_stage(paths: Paths, res: StageResult) -> None:
    paths.state.mkdir(parents=True, exist_ok=True)
    (paths.state / f"{res.name}.json").write_text(
        json.dumps(res.__dict__, indent=2), encoding="utf-8")


def stage_done(paths: Paths, name: str) -> bool:
    f = paths.state / f"{name}.json"
    if not f.exists():
        return False
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


def read_stages(paths: Paths) -> list[dict]:
    if not paths.state.exists():
        return []
    out = []
    for f in sorted(paths.state.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def skip_note(paths: Paths, name: str, reason: str) -> None:
    """Record a stage that was deliberately not run. RESULTS.md prints these verbatim."""
    write_stage(paths, StageResult(name, 0.0, 0.0, 0.0, "skipped", {"reason": reason}))
    log(f"[{name}] SKIPPED — {reason}")


# --------------------------------------------------------------------------- misc
def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, log_name: str | None = None,
        check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run an external tool, tee its output to logs/, raise on failure."""
    LOGS.mkdir(parents=True, exist_ok=True)
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"  $ {printable}")
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True, errors="replace", env=e)
    if log_name:
        (LOGS / f"{log_name}.log").write_text(
            f"$ {printable}\n\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}",
            encoding="utf-8")
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-25:]
        log(f"  ! exit {p.returncode}\n     " + "\n     ".join(tail))
        if check:
            raise RuntimeError(f"{cmd[0]} failed with exit {p.returncode}")
    return p


def which(*names: str) -> str | None:
    """Find a tool on PATH or in tools/ (where we unzip portable binaries)."""
    for n in names:
        hit = shutil.which(n)
        if hit:
            return hit
    for n in names:
        exe = n if n.lower().endswith(".exe") else n + ".exe"
        for cand in TOOLS.rglob(exe):
            if cand.is_file():
                return str(cand)
    return None


IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".JPG", ".JPEG", ".PNG"}


def list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}])


def mb(p: Path) -> float:
    return round(p.stat().st_size / 1024**2, 2) if p.exists() else 0.0


def base_argparser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--dataset", required=True, help="name under data/")
    ap.add_argument("--force", action="store_true", help="rerun even if checkpointed")
    return ap
