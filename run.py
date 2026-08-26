"""MANAH PoC — chains every pipeline stage.

    python run.py --dataset gingerbread
    python run.py --dataset own_dish            # same command, zero code changes

Each stage is a standalone script under pipeline/ and is also runnable alone.
Completed stages are checkpointed in work/<ds>/.state and skipped on rerun
unless --force is passed. A stage that fails is logged and the chain stops
there unless --keep-going is set; the report stage always runs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import Paths, log, read_stages  # noqa: E402

STAGES = [
    ("s0_ingest",       "ingest + downscale"),
    ("s1_masks",        "SAM 2.1 subject isolation"),
    ("s2_colmap",       "sparse reconstruction, masked vs unmasked"),
    ("s3_mvs",          "OpenMVS densify + mesh + texture (path A)"),
    ("s3b_realityscan", "RealityScan CLI (path B, comparison)"),
    ("s4_cleanup",      "plane cut, component filter, decimate"),
    ("s5_export",       "GLB / USDZ / optimise"),
    ("s5b_render",      "turntable screenshots"),
    ("s6_report",       "benchmark.json + RESULTS.md + viewer"),
]


def python_exe() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the MANAH reconstruction pipeline")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force", action="store_true", help="rerun all stages")
    ap.add_argument("--from", dest="start", help="start at this stage")
    ap.add_argument("--only", help="run just this stage")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing stage")
    args, extra = ap.parse_known_args()

    p = Paths(args.dataset).ensure()
    todo = [s for s in STAGES]
    if args.only:
        todo = [s for s in STAGES if s[0] == args.only]
        if not todo:
            sys.exit(f"unknown stage {args.only}; known: {[s[0] for s in STAGES]}")
    elif args.start:
        names = [s[0] for s in STAGES]
        if args.start not in names:
            sys.exit(f"unknown stage {args.start}; known: {names}")
        todo = STAGES[names.index(args.start):]

    t0 = time.perf_counter()
    failed = []
    for name, desc in todo:
        script = ROOT / "pipeline" / f"{name}.py"
        if not script.exists():
            log(f"[{name}] script missing — skipping ({desc})")
            continue
        cmd = [python_exe(), str(script), "--dataset", args.dataset] + extra
        if args.force:
            cmd.append("--force")
        log(f"=== {name}: {desc} ===")
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            failed.append(name)
            log(f"!!! {name} exited {rc}")
            if name == "s6_report" or args.keep_going:
                continue
            log("stopping. rerun with --keep-going to push past this, "
                "or --from <stage> after fixing it.")
            # still produce the report so the failure is documented
            rep = ROOT / "pipeline" / "s6_report.py"
            if rep.exists() and name != "s6_report":
                subprocess.run([python_exe(), str(rep), "--dataset", args.dataset,
                                "--force"], cwd=str(ROOT))
            sys.exit(rc)

    total = round(time.perf_counter() - t0, 1)
    log(f"=== pipeline finished in {total}s ===")
    for s in read_stages(p):
        log(f"    {s['name']:<18} {s['status']:<8} {s['seconds']:>8}s "
            f"ram={s.get('peak_ram_mb', 0)}MB vram={s.get('peak_vram_mb', 0)}MB")
    if failed:
        log(f"failed stages: {failed}")


if __name__ == "__main__":
    main()
