"""Stage 3 (path A) — OpenMVS densify + mesh.

Texturing deliberately does NOT happen here; it happens in s5_export, after
s4_cleanup has cut the table plane and decimated. Texturing last means UVs are
generated on the final geometry, which sidesteps the usual "decimate without
destroying UVs" problem entirely.

Masks reach OpenMVS as <stem>.mask.png in --mask-path with --ignore-mask-label 0,
so background pixels are dropped during depth-map estimation. That is the step
that actually keeps the table out of the model — masking only at the mesh stage
would already have baked table geometry into the depth maps.

All flags below were read out of `DensifyPointCloud --help` / `ReconstructMesh
--help` on the exact v2.4.0 binaries in tools/ (OpenMVS writes help to a .log
file rather than stdout).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Paths, Stage, base_argparser, device, log, run, stage_done, which

MVS_TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
             "RefineMesh", "TextureMesh")


def mvs_bin(name: str) -> str:
    exe = which(name)
    if not exe:
        raise RuntimeError(f"{name}.exe not found in tools/ — see SETUP.md")
    return exe


def peak_mb_from_log(work: Path, tool: str) -> float | None:
    """OpenMVS prints its own PeakWorkingSetSize; cheaper and more accurate than
    sampling the child from outside."""
    logs = sorted(work.glob(f"{tool}-*.log"), key=lambda f: f.stat().st_mtime)
    if not logs:
        return None
    txt = logs[-1].read_text(encoding="utf-8", errors="replace")
    m = re.search(r"PeakWorkingSetSize\s+([\d.]+)(MB|GB|KB)", txt)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return round(v * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[unit], 1)


def face_count(ply: Path) -> int:
    """Read the face count out of the PLY header without loading the mesh."""
    try:
        with open(ply, "rb") as f:
            for _ in range(60):
                line = f.readline().decode("ascii", "replace")
                if line.startswith("element face"):
                    return int(line.split()[-1])
                if line.startswith("end_header"):
                    break
    except Exception:
        pass
    return 0


def main() -> None:
    ap = base_argparser("Stage 3A: OpenMVS densify + mesh")
    ap.add_argument("--resolution-level", type=int, default=None,
                    help="image downscale for depth maps (0=full, 1=half, 2=quarter). "
                         "Default 2 on CPU, 1 on CUDA.")
    ap.add_argument("--number-views", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2,
                    help="PatchMatch iterations on CPU (OpenMVS default 3)")
    ap.add_argument("--geometric-iters", type=int, default=0,
                    help="geometric-consistency iterations on CPU (OpenMVS default 2)")
    ap.add_argument("--target-faces", type=int, default=0,
                    help="0 = keep full mesh; s4_cleanup does the web decimation")
    ap.add_argument("--no-masks", action="store_true")
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s3_mvs") and not args.force:
        log("[s3_mvs] already done (use --force to redo)")
        return

    und = p.work / "undistorted"
    if not (und / "images").exists():
        raise RuntimeError(f"{und}/images missing — run s2_colmap first")

    dev = device()
    rlevel = args.resolution_level if args.resolution_level is not None \
        else (1 if dev == "cuda" else 2)

    mvs = p.mvs
    mvs.mkdir(parents=True, exist_ok=True)
    scene = mvs / "scene.mvs"
    dense = mvs / "scene_dense.mvs"
    mesh_mvs = mvs / "scene_mesh.mvs"
    mesh_ply = mvs / "scene_mesh.ply"

    masks = p.work / "mvs_masks"
    use_masks = (not args.no_masks) and masks.exists() and any(masks.glob("*.mask.png"))

    with Stage(p, "s3_mvs") as st:
        st.notes["resolution_level"] = rlevel
        st.notes["masks_used"] = bool(use_masks)
        if not use_masks:
            st.notes["mask_warning"] = ("running WITHOUT masks — the table will be "
                                        "reconstructed; s4_cleanup must remove it")
            log("  ! no OpenMVS masks found — densifying unmasked")

        # 1. COLMAP -> MVS scene
        run([mvs_bin("InterfaceCOLMAP"),
             "-i", und,
             "-o", scene,
             "--image-folder", und / "images",
             "-w", mvs],
            cwd=mvs, log_name="mvs_interface")
        st.notes["peak_mb_interface"] = peak_mb_from_log(mvs, "InterfaceCOLMAP")

        # 2. Dense point cloud. --tower-mode 0 matters: the default (4) injects a
        #    synthetic cylinder of points at the ROI centre, which is wrong for a
        #    single object and shows up as a spike through the middle of the dish.
        cmd = [mvs_bin("DensifyPointCloud"),
               "-i", scene,
               "-o", dense,
               "-w", mvs,
               "--resolution-level", str(rlevel),
               "--number-views", str(args.number_views),
               "--max-threads", str(os.cpu_count() or 0),
               "--tower-mode", "0",
               "--estimate-roi", "1.1",
               "--crop-to-roi", "1",
               # CPU PatchMatch cost control. The defaults (iters 3,
               # geometric-iters 2, sub-resolution-levels 2) are tuned for a GPU
               # and run for hours on a 15 W laptop part that thermally throttles
               # partway through. On CUDA the defaults are restored.
               "--iters", "3" if dev == "cuda" else str(args.iters),
               "--geometric-iters", "2" if dev == "cuda" else str(args.geometric_iters),
               "--sub-resolution-levels", "2" if dev == "cuda" else "1"]
        if use_masks:
            # OpenMVS concatenates this string directly onto the image basename
            # (strMaskPath + getFileName(...)), so a missing trailing separator
            # silently produces the wrong filename and every mask is ignored --
            # which looks exactly like "masking didn't work" rather than an error.
            mask_arg = str(masks) + os.sep
            cmd += ["--mask-path", mask_arg, "--ignore-mask-label", "0"]
            st.notes["mask_path_arg"] = mask_arg
        run(cmd, cwd=mvs, log_name="mvs_densify")
        st.notes["peak_mb_densify"] = peak_mb_from_log(mvs, "DensifyPointCloud")

        dense_ply = mvs / "scene_dense.ply"
        st.notes["dense_points_ply_mb"] = round(dense_ply.stat().st_size / 1024**2, 2) \
            if dense_ply.exists() else 0

        # 3. Mesh. crop-to-roi keeps the graph-cut inside the estimated object ROI.
        run([mvs_bin("ReconstructMesh"),
             "-i", dense,
             "-o", mesh_mvs,
             "-w", mvs,
             "--export-type", "ply",
             "--target-face-num", str(args.target_faces),
             "--remove-spurious", "20",
             "--remove-spikes", "1",
             "--close-holes", "30",
             "--smooth", "2",
             "--crop-to-roi", "1",
             "--max-threads", str(os.cpu_count() or 0)],
            cwd=mvs, log_name="mvs_mesh")
        st.notes["peak_mb_mesh"] = peak_mb_from_log(mvs, "ReconstructMesh")

        if not mesh_ply.exists():
            cands = sorted(mvs.glob("scene_mesh*.ply"))
            if cands:
                shutil.copy(cands[0], mesh_ply)
        if not mesh_ply.exists():
            raise RuntimeError(f"ReconstructMesh produced no ply in {mvs}")

        tris = face_count(mesh_ply)
        st.notes["tris"] = tris
        st.notes["mesh_ply"] = str(mesh_ply)
        st.notes["note"] = (f"resolution-level {rlevel}, "
                            f"{'masked' if use_masks else 'UNMASKED'}, {tris} tris")
        log(f"  mesh: {mesh_ply.name}, {tris} triangles")


if __name__ == "__main__":
    main()
