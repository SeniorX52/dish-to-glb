"""Stage 5 — texture the cleaned mesh, then export web assets.

Order: TextureMesh (UVs generated on the FINAL, already-decimated geometry)
-> Blender -> GLB + USDZ -> gltf-transform optimise.

Two non-obvious choices, both from verified behaviour rather than defaults:

  * `gltf-transform optimize` defaults to --compress meshopt, but <model-viewer>
    does not enable a meshopt decoder by default, so the bare default silently
    produces a model that will not load. Draco is used instead: model-viewer
    auto-fetches a Draco decoder with zero client configuration.
  * `optimize` also defaults --simplify true, which decimates geometry behind
    your back. For a scan that is not acceptable, so it is turned off; s4_cleanup
    already decided the triangle budget deliberately.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, Paths, Stage, base_argparser, log, mb, run, stage_done,
                    which)

GLTF_DIR = ROOT / "tools" / "gltf"


def blender_exe() -> str | None:
    hit = which("blender")
    if hit:
        return hit
    roots = [Path(r"C:\Program Files\Blender Foundation"),
             Path(r"C:\Program Files (x86)\Blender Foundation")]
    cands = []
    for r in roots:
        if r.exists():
            cands += list(r.rglob("blender.exe"))
    return str(sorted(cands)[-1]) if cands else None


def gltf_transform(args: list[str], log_name: str) -> subprocess.CompletedProcess:
    """Use the LOCAL install: a global @gltf-transform/cli on Windows resolves two
    conflicting sharp/libvips builds and every texture command dies with
    'colourspace: parameter space not set'. tools/gltf pins a single sharp."""
    cli = GLTF_DIR / "node_modules" / "@gltf-transform" / "cli" / "bin" / "cli.js"
    if not cli.exists():
        raise RuntimeError(f"{cli} missing — run `npm install` in {GLTF_DIR}")
    return run(["node", str(cli)] + args, cwd=GLTF_DIR, log_name=log_name, check=False)


def texture_mesh(p: Paths, mesh: Path, notes: dict, max_texture: int = 4096) -> Path:
    """OpenMVS TextureMesh on the cleaned mesh.

    Two flag traps here, both verified against `TextureMesh --help` on the v2.4.0
    binary rather than copied from the densify step:

      * `--ignore-mask-label` has DIFFERENT sentinels to DensifyPointCloud. Here
        the default -1 means "auto estimate mask for lens distortion" and -2
        means disabled. Copying densify's -1 across would silently swap our
        object masks for a lens-distortion mask, letting table pixels bleed into
        the atlas at the silhouette. 0 is what honours our masks.
      * `--max-texture-size` defaults to 8192. Generating an 8K atlas on a
        laptop CPU is slow and pointless when the web target downsizes to 2048.
    """
    from s3_mvs import mvs_bin, peak_mb_from_log

    out = p.mvs / "model_textured.mvs"
    for stale in p.mvs.glob("model_textured*"):
        stale.unlink(missing_ok=True)

    cmd = [mvs_bin("TextureMesh"),
           "-i", p.mvs / "scene_dense.mvs",
           "--mesh-file", mesh,
           "-o", out,
           "-w", p.mvs,
           "--export-type", "obj",
           "--resolution-level", "0",          # texture from the full working images
           "--texture-size-multiple", "0",     # power of two
           "--max-texture-size", str(max_texture),
           "--close-holes", "30",
           # OpenMVS v2.4.0 regression (issue #1251): a wrong template parameter
           # in the seam-leveling image sampler (Types.inl ~L2401) corrupts
           # sampled pixels, covering ~25-35% of every texture patch in pure
           # black; the model renders near-black. Both global AND local passes
           # are affected (measured here: 0.36 / 0.24 dark fraction vs 0.013
           # with both off). Fixed in develop commit eeedab7a but in no release
           # yet. Cost of the workaround: exposure seams between patches are
           # not blended. On a CUDA machine, v2.3.0's TextureMesh (CUDA-only
           # Windows build) or a develop build restores seam leveling.
           "--global-seam-leveling", "0",
           "--local-seam-leveling", "0",
           "--max-threads", str(os.cpu_count() or 0)]
    masks = p.work / "mvs_masks"
    if masks.exists() and any(masks.glob("*.mask.png")):
        cmd += ["--ignore-mask-label", "0"]
        notes["texture_masked"] = True
    else:
        cmd += ["--ignore-mask-label", "-2"]   # explicitly disabled, not "auto"
        notes["texture_masked"] = False
    run(cmd, cwd=p.mvs, log_name="mvs_texture")
    notes["peak_mb_texture"] = peak_mb_from_log(p.mvs, "TextureMesh")

    objs = sorted(p.mvs.glob("model_textured*.obj"),
                  key=lambda f: f.stat().st_mtime)
    if not objs:
        raise RuntimeError(f"TextureMesh produced no .obj in {p.mvs}")
    obj = objs[-1]
    tex = sorted(list(p.mvs.glob("model_textured*.png")) +
                 list(p.mvs.glob("model_textured*.jpg")))
    notes["texture_files"] = [t.name for t in tex]
    notes["texture_mb"] = round(sum(t.stat().st_size for t in tex) / 1024**2, 2)
    return obj


def main() -> None:
    ap = base_argparser("Stage 5: texture + GLB/USDZ export")
    ap.add_argument("--texture-size", type=int, default=2048,
                    help="final web atlas size after gltf-transform")
    ap.add_argument("--max-texture", type=int, default=4096,
                    help="OpenMVS atlas size before web downsizing (default 8192 is wasteful)")
    ap.add_argument("--compress", choices=["draco", "meshopt", "quantize", "false"],
                    default="draco")
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s5_export") and not args.force:
        log("[s5_export] already done (use --force to redo)")
        return

    mesh = p.mvs / "scene_clean.ply"
    if not mesh.exists():
        mesh = p.mvs / "scene_mesh.ply"
        log(f"  ! scene_clean.ply missing, falling back to {mesh.name}")
    if not mesh.exists():
        raise RuntimeError("no mesh to texture — run s3_mvs/s4_cleanup first")

    caveats: list[str] = []
    with Stage(p, "s5_export") as st:
        # ---------------------------------------------------------- texture
        obj = texture_mesh(p, mesh, st.notes, args.max_texture)
        st.notes["textured_obj"] = str(obj)
        log(f"  textured -> {obj.name} ({mb(obj)} MB)")

        # keep a copy of the raw OBJ set beside the web assets
        for f in p.mvs.glob(obj.stem + "*"):
            if f.suffix.lower() in (".obj", ".mtl", ".png", ".jpg"):
                shutil.copy(f, p.out / f.name.replace(obj.stem, "model"))

        # ---------------------------------------------------------- GLB/USDZ
        raw_glb = p.out / "model_raw.glb"
        usdz = p.out / "model.usdz"
        blender = blender_exe()
        if blender:
            st.notes["blender"] = blender
            r = run([blender, "--background", "--factory-startup",
                     "--python-exit-code", "1",
                     "--python", ROOT / "pipeline" / "blender_convert.py",
                     "--", str(obj), str(raw_glb), str(usdz)],
                    log_name="blender_convert", check=False)
            m = re.search(r"BLENDER_RESULT_JSON (\{.*\})", (r.stdout or "") + (r.stderr or ""))
            if m:
                info = json.loads(m.group(1))
                st.notes["blender_result"] = info
                if info.get("usdz_error"):
                    caveats.append(f"USDZ export failed in Blender: {info['usdz_error']}")
            if not raw_glb.exists():
                caveats.append("Blender did not produce a GLB; falling back to obj2gltf")
        else:
            caveats.append("Blender not found; using obj2gltf and skipping USDZ")

        # ---------------------------------------------------------- fallback GLB
        if not raw_glb.exists():
            cli = GLTF_DIR / "node_modules" / "obj2gltf" / "bin" / "obj2gltf.js"
            if cli.exists():
                run(["node", str(cli), "-i", str(obj), "-o", str(raw_glb), "-b"],
                    cwd=GLTF_DIR, log_name="obj2gltf", check=False)
        if not raw_glb.exists():
            raise RuntimeError("no GLB produced by Blender or obj2gltf")
        st.notes["raw_glb_mb"] = mb(raw_glb)

        # ---------------------------------------------------------- optimise
        final_glb = p.out / "model.glb"
        r = gltf_transform([
            "optimize", str(raw_glb), str(final_glb),
            "--compress", args.compress,          # draco: model-viewer needs no config
            "--texture-compress", "webp",
            "--texture-size", str(args.texture_size),
            "--simplify", "false",                # do NOT silently decimate a scan
            "--prune-solid-textures", "false",
        ], "gltf_optimize")
        if not final_glb.exists():
            caveats.append("gltf-transform optimize failed; shipping the uncompressed GLB. "
                           "stderr tail: " + (r.stderr or "")[-300:])
            shutil.copy(raw_glb, final_glb)
            st.notes["optimized"] = False
        else:
            st.notes["optimized"] = True
            st.notes["compression"] = args.compress

        caveats.append(
            "Texture seams between patches are not blended: OpenMVS v2.4.0's "
            "seam-leveling is disabled to work around the black-patch regression "
            "(github.com/cdcseacave/openMVS issue #1251, fixed in develop commit "
            "eeedab7a, no release yet). On a CUDA machine, use a develop build or "
            "v2.3.0's TextureMesh to restore seam blending.")
        st.notes["glb_mb"] = mb(final_glb)
        st.notes["usdz_mb"] = mb(usdz)
        st.notes["raw_to_final_ratio"] = (
            round(mb(raw_glb) / mb(final_glb), 2) if mb(final_glb) else None)

        # ---------------------------------------------------------- verify
        insp = gltf_transform(["inspect", str(final_glb)], "gltf_inspect")
        text = (insp.stdout or "") + (insp.stderr or "")
        st.notes["extensions"] = sorted(set(re.findall(
            r"(KHR_[A-Za-z_]+|EXT_[A-Za-z_]+)", text)))
        if not usdz.exists():
            caveats.append("No USDZ produced — see blender_convert log.")
        if caveats:
            st.notes["caveats"] = caveats
        log(f"  GLB {st.notes['glb_mb']} MB (from {st.notes['raw_glb_mb']} MB), "
            f"USDZ {st.notes['usdz_mb']} MB, ext={st.notes['extensions']}")


if __name__ == "__main__":
    main()
