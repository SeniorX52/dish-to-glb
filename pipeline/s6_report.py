"""Stage 6 — benchmark.json + RESULTS.md + the static viewer.

Hard rule: every number written here is read back out of a stage record that
the code actually measured. Nothing is estimated. A stage that did not run is
printed as "skipped" with its recorded reason, never as a guess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, Paths, Stage, base_argparser, host_info, log, mb,
                    read_stages)

ORDER = ["s0_ingest", "s1_masks", "s2_colmap", "s3_mvs", "s3b_realityscan",
         "s4_cleanup", "s5_export", "s5b_render", "s6_report"]


def _sorted_stages(stages: list[dict]) -> list[dict]:
    return sorted(stages, key=lambda s: ORDER.index(s["name"])
                  if s["name"] in ORDER else 99)


def _note(stages: list[dict], stage: str, key: str, default=None):
    for s in stages:
        if s["name"] == stage:
            return s.get("notes", {}).get(key, default)
    return default


def _status(stages: list[dict], stage: str) -> str:
    for s in stages:
        if s["name"] == stage:
            return s["status"]
    return "not-run"


def build_benchmark(p: Paths, stages: list[dict]) -> dict:
    host = host_info()
    ing = next((s for s in stages if s["name"] == "s0_ingest"), {})
    notes = ing.get("notes", {})
    glb = p.out / "model.glb"
    usdz = p.out / "model.usdz"
    obj = p.out / "model.obj"

    ran = [s for s in stages if s["status"] == "ok"]
    bench = {
        "dataset": p.ds,
        "n_images": notes.get("n_images", 0),
        "resolution": ", ".join(notes.get("source_resolutions", [])) or "unknown",
        "working_resolution_max_dim": notes.get("max_dim"),
        "camera": notes.get("cameras", {}),
        "gpu": host.get("gpu", "unknown"),
        "cuda_available": host.get("cuda", False),
        "cpu": host.get("cpu", "unknown"),
        "cpu_threads": host.get("cpu_threads"),
        "ram_gb": host.get("ram_gb"),
        "os": host.get("os"),
        "torch": host.get("torch", "not-installed"),
        "stages": [
            {
                "name": s["name"],
                "seconds": s["seconds"],
                "peak_vram_mb": s.get("peak_vram_mb", 0),
                "peak_ram_mb": s.get("peak_ram_mb", 0),
                "status": s["status"],
                **({"reason": s["notes"]["reason"]} if s.get("notes", {}).get("reason") else {}),
            }
            for s in _sorted_stages(stages)
        ],
        "total_seconds": round(sum(s["seconds"] for s in ran), 2),
        "outputs": {
            "glb_mb": mb(glb),
            "usdz_mb": mb(usdz),
            "obj_mb": mb(obj),
            "tris": _note(stages, "s4_cleanup", "tris_after",
                          _note(stages, "s3_mvs", "tris", 0)),
        },
    }
    return bench


def _img_block(p: Paths, name: str, caption: str) -> str:
    # RESULTS.md lives at the repo root; reference images relative to it so
    # they render on GitHub, not just in editors that resolve loosely.
    f = p.out / name
    rel = f"outputs/{p.ds}/{name}"
    return f"\n**{caption}**\n\n![{caption}]({rel})\n" if f.exists() else \
           f"\n**{caption}** — not produced.\n"


def build_results(p: Paths, stages: list[dict], bench: dict) -> str:
    L = []
    a = L.append
    a(f"# RESULTS — `{p.ds}`\n")
    a("> Every number in this file was measured by the pipeline and read back out of\n"
      "> `work/<dataset>/.state/*.json`. Nothing here is estimated. Stages that did not\n"
      "> run are printed as `skipped` with the recorded reason.\n")

    a("\n## Machine actually used\n")
    a("| | |\n|---|---|")
    a(f"| CPU | {bench['cpu']} ({bench['cpu_threads']} threads) |")
    a(f"| GPU | {bench['gpu']} |")
    a(f"| CUDA available | **{bench['cuda_available']}** |")
    a(f"| RAM | {bench['ram_gb']} GB |")
    a(f"| OS | {bench['os']} |")
    a(f"| torch | {bench['torch']} |")

    a("\n## Dataset\n")
    a("| | |\n|---|---|")
    a(f"| name | `{p.ds}` |")
    a(f"| images | {bench['n_images']} |")
    a(f"| source resolution | {bench['resolution']} |")
    a(f"| working max dimension | {bench['working_resolution_max_dim']} px |")
    cams = ", ".join(f"{k} ×{v}" for k, v in (bench.get("camera") or {}).items())
    a(f"| camera (EXIF) | {cams or 'not present in EXIF'} |")

    a("\n## Stage timings\n")
    a("| stage | status | seconds | peak RAM MB | peak VRAM MB | note |")
    a("|---|---|---:|---:|---:|---|")
    for s in bench["stages"]:
        a(f"| `{s['name']}` | {s['status']} | {s['seconds']} | {s['peak_ram_mb']} "
          f"| {s['peak_vram_mb']} | {s.get('reason', '')} |")
    a(f"\n**Total measured wall clock: {bench['total_seconds']} s "
      f"({round(bench['total_seconds'] / 60, 1)} min)** across stages that ran.\n")

    # ---- masked vs unmasked alignment
    a("\n## Sparse alignment: masked vs unmasked\n")
    cmp = _note(stages, "s2_colmap", "comparison")
    if cmp:
        a("| variant | registered images | of | mean reprojection error px | points | seconds |")
        a("|---|---:|---:|---:|---:|---:|")
        for v in cmp:
            a(f"| {v['variant']} | {v.get('registered', '—')} | {v.get('n_images', '—')} "
              f"| {v.get('mean_reproj_err', '—')} | {v.get('points', '—')} "
              f"| {v.get('seconds', '—')} |")
        a(f"\n**Winner used downstream: `{_note(stages, 's2_colmap', 'winner', '—')}`** "
          f"— {_note(stages, 's2_colmap', 'winner_reason', '')}\n")
    else:
        a(f"Not produced — `s2_colmap` status: **{_status(stages, 's2_colmap')}**.\n")

    # ---- path A vs path B
    a("\n## Path A (COLMAP + OpenMVS) vs Path B (RealityScan)\n")
    a("| | path A | path B |")
    a("|---|---|---|")
    a(f"| status | {_status(stages, 's3_mvs')} | {_status(stages, 's3b_realityscan')} |")
    a(f"| seconds | {next((s['seconds'] for s in bench['stages'] if s['name'] == 's3_mvs'), '—')} "
      f"| {next((s['seconds'] for s in bench['stages'] if s['name'] == 's3b_realityscan'), '—')} |")
    b_reason = _note(stages, "s3b_realityscan", "reason", "")
    a(f"| note | {_note(stages, 's3_mvs', 'note', '')} | {b_reason} |")

    a("\n## Outputs\n")
    a("| artefact | size MB |\n|---|---:|")
    for k, v in bench["outputs"].items():
        if k != "tris":
            a(f"| {k.replace('_mb', '')} | {v} |")
    a(f"\nFinal triangle count: **{bench['outputs']['tris']}**\n")

    a("\n## Visual checks\n")
    a(_img_block(p, "mask_check.jpg", "Mask contact sheet (12 evenly spaced frames)"))
    for i in (1, 2, 3):
        a(_img_block(p, f"render_{i}.png", f"Render {i}"))

    a("\n## What failed or looks bad\n")
    bad = [s for s in bench["stages"] if s["status"] in ("failed", "skipped")]
    if not bad:
        a("- Nothing failed or was skipped in this run.\n")
    for s in bad:
        why = s.get("reason") or _note(stages, s["name"], "error", "no reason recorded")
        a(f"- **`{s['name']}` — {s['status']}**: {why}")
    for extra in (_note(stages, "s4_cleanup", "caveats", []) or []):
        a(f"- {extra}")
    for extra in (_note(stages, "s5_export", "caveats", []) or []):
        a(f"- {extra}")

    a("\n## Reproduce\n")
    a("```\npython run.py --dataset " + p.ds + "\n```\n")
    a(f"Viewer: open `viewer/index.html?dataset={p.ds}` in a browser "
      "(it loads `outputs/<dataset>/model.glb`).\n")
    return "\n".join(L) + "\n"


def main() -> None:
    args = base_argparser("Stage 6: benchmark + report").parse_args()
    p = Paths(args.dataset).ensure()
    with Stage(p, "s6_report") as st:
        stages = [s for s in read_stages(p) if s["name"] != "s6_report"]
        bench = build_benchmark(p, stages)
        (p.out / "benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
        (ROOT / "RESULTS.md").write_text(build_results(p, stages, bench), encoding="utf-8")
        st.notes["stages_recorded"] = len(stages)
        st.notes["total_seconds"] = bench["total_seconds"]
        log(f"  wrote outputs/{p.ds}/benchmark.json and RESULTS.md "
            f"({len(stages)} stages, {bench['total_seconds']}s total)")


if __name__ == "__main__":
    main()
