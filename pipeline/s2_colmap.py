"""Stage 2 — sparse reconstruction with COLMAP, run twice: masked and unmasked.

The brief's hypothesis is that masking during feature extraction can starve
alignment on a low-texture object, and that the right configuration is to align
on the full image and mask only from densification onward. This stage tests
that empirically instead of assuming it: both variants are built, both are
measured with `colmap model_analyzer`, and the winner is chosen on registered
image count with mean reprojection error as tie-break.

Flags here were read out of `colmap.exe <cmd> --help` on the exact 4.1.1 binary
in tools/, not from a tutorial. Note 4.1.1 renamed the classic SiftExtraction
GPU switches to FeatureExtraction.use_gpu / FeatureMatching.use_gpu while
keeping SiftExtraction.* for SIFT-specific tuning.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Paths, Stage, base_argparser, list_images, log, run,
                    stage_done, which)

USE_GPU = "0"  # set from device() at runtime; CPU SIFT on this box


def colmap_bin() -> str:
    exe = which("colmap")
    if not exe:
        raise RuntimeError("colmap.exe not found on PATH or in tools/")
    return exe


# ---------------------------------------------------------------- mask plumbing
def write_colmap_masks(p: Paths) -> Path:
    """COLMAP wants <image_filename>.png inside --ImageReader.mask_path, with
    zero-valued pixels excluded. We also drop a <stem>.png alias so the lookup
    succeeds under either convention."""
    src = p.masks
    dst = p.work / "masks_colmap"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for m in sorted(src.glob("*.png")):
        img = Image.open(m).convert("L")
        # Size must match the image on disk EXACTLY. COLMAP's MaskFeatures()
        # treats an out-of-bounds pixel lookup the same as a black pixel and
        # deletes the keypoint, with no warning — a mismatched mask silently
        # deletes every feature and looks like "alignment just failed".
        photo = p.resized / f"{m.stem}.jpg"
        if photo.exists():
            with Image.open(photo) as ph:
                if ph.size != img.size:
                    raise RuntimeError(
                        f"mask/image size mismatch for {m.stem}: mask {img.size} "
                        f"vs image {ph.size}. COLMAP would silently drop every "
                        f"keypoint. Rerun s0_ingest and s1_masks together.")
        img.save(dst / f"{m.stem}.jpg.png")   # documented convention (appended .png)
        img.save(dst / f"{m.stem}.png")       # 4.1.1 fallback spelling
        n += 1
    if n == 0:
        raise RuntimeError(f"no masks in {src}")
    log(f"  prepared {n} COLMAP masks in {dst}")
    return dst


def masks_as_images(p: Paths) -> Path:
    """A folder of masks named like the images, so image_undistorter can be run
    over them with identical camera parameters. Undistorting the masks is the
    only way to keep them aligned with the undistorted images OpenMVS consumes."""
    dst = p.work / "masks_as_images"
    dst.mkdir(parents=True, exist_ok=True)
    for m in sorted(p.masks.glob("*.png")):
        # 3-channel: image_undistorter expects real images, not 1-channel data.
        # Quality 100 keeps the 0/255 step crisp enough to re-threshold at 127.
        Image.open(m).convert("L").convert("RGB").save(
            dst / f"{m.stem}.jpg", "JPEG", quality=100)
    return dst


# ---------------------------------------------------------------- colmap steps
def sparse(p: Paths, variant: str, matcher: str, max_features: int) -> dict:
    """Build one sparse model. variant is 'masked' or 'unmasked'."""
    import time
    t0 = time.perf_counter()
    exe = colmap_bin()
    ws = p.colmap(variant)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    (ws / "sparse").mkdir(parents=True, exist_ok=True)
    db = ws / "database.db"

    fe = [exe, "feature_extractor",
          "--database_path", db,
          "--image_path", p.resized,
          "--ImageReader.single_camera", "1",
          "--ImageReader.camera_model", "SIMPLE_RADIAL",
          "--FeatureExtraction.use_gpu", USE_GPU,
          "--SiftExtraction.max_num_features", str(max_features)]
    if variant == "masked":
        fe += ["--ImageReader.mask_path", write_colmap_masks(p)]
    run(fe, log_name=f"colmap_{variant}_extract")

    if matcher == "sequential":
        run([exe, "sequential_matcher", "--database_path", db,
             "--FeatureMatching.use_gpu", USE_GPU,
             "--SequentialMatching.overlap", "15",
             "--SequentialMatching.quadratic_overlap", "1",
             "--SequentialMatching.loop_detection", "0"],
            log_name=f"colmap_{variant}_match")
    else:
        run([exe, "exhaustive_matcher", "--database_path", db,
             "--FeatureMatching.use_gpu", USE_GPU],
            log_name=f"colmap_{variant}_match")

    run([exe, "mapper",
         "--database_path", db,
         "--image_path", p.resized,
         "--output_path", ws / "sparse"],
        log_name=f"colmap_{variant}_mapper")

    stats = analyze(p, ws, variant)
    stats.update(variant=variant, seconds=round(time.perf_counter() - t0, 2),
                 n_images=len(list_images(p.resized)))
    return stats


NUM = r"([-\d.]+)"
PATTERNS = {
    "cameras":        re.compile(r"Cameras:\s*" + NUM),
    "images":         re.compile(r"Images:\s*" + NUM),
    "registered":     re.compile(r"Registered images:\s*" + NUM),
    "points":         re.compile(r"Points:\s*" + NUM),
    "observations":   re.compile(r"Observations:\s*" + NUM),
    "mean_track_len": re.compile(r"Mean track length:\s*" + NUM),
    "mean_reproj_err": re.compile(r"Mean reprojection error:\s*" + NUM),
}


def analyze(p: Paths, ws: Path, variant: str) -> dict:
    """Read registered-image count and mean reprojection error out of COLMAP."""
    models = sorted([d for d in (ws / "sparse").iterdir() if d.is_dir()]) \
        if (ws / "sparse").exists() else []
    if not models:
        return {"registered": 0, "mean_reproj_err": None, "points": 0,
                "model_dir": None, "note": "mapper produced no model"}
    # Largest sub-model wins; COLMAP splits when the scene does not connect.
    best, best_stats = None, None
    for m in models:
        r = run([colmap_bin(), "model_analyzer", "--path", m],
                log_name=f"colmap_{variant}_analyze_{m.name}", check=False)
        text = (r.stderr or "") + (r.stdout or "")
        s = {}
        for k, rx in PATTERNS.items():
            hit = rx.search(text)
            if hit:
                try:
                    s[k] = float(hit.group(1))
                except ValueError:
                    pass
        s["registered"] = int(s.get("registered", 0))
        s["points"] = int(s.get("points", 0))
        if best_stats is None or s["registered"] > best_stats["registered"]:
            best, best_stats = m, s
    best_stats["model_dir"] = str(best)
    best_stats["n_submodels"] = len(models)
    if best_stats.get("mean_reproj_err") is not None:
        best_stats["mean_reproj_err"] = round(best_stats["mean_reproj_err"], 4)
    return best_stats


def undistort(p: Paths, model_dir: Path, image_path: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    run([colmap_bin(), "image_undistorter",
         "--image_path", image_path,
         "--input_path", model_dir,
         "--output_path", out,
         "--output_type", "COLMAP"],
        log_name=f"colmap_undistort_{out.name}")
    return out


def build_mvs_masks(p: Paths, undist_masks: Path, dst: Path) -> int:
    """Threshold the undistorted masks back to binary and name them the way
    OpenMVS actually looks them up.

    OpenMVS builds the mask path as `strMaskPath + Util::getFileName(image)`,
    and getFileName strips at the last '.', so for `20201115_115903.jpg` it wants
    `20201115_115903.mask.png` — extension replaced, not appended. Exactly one
    convention per directory: a stale second spelling is how you end up debugging
    a mask that is present on disk and still ignored.

    DensifyPointCloud hard-fails if any single mask is missing from --mask-path,
    so this must emit one per undistorted image.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.png"):
        stale.unlink()
    n = 0
    for f in sorted((undist_masks / "images").glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        a = np.array(Image.open(f).convert("L"))
        solid = solidify_mask(np.where(a > 127, 255, 0).astype(np.uint8))
        Image.fromarray(solid).save(dst / f"{f.stem}.mask.png")
        n += 1
    return n


def solidify_mask(m: "np.ndarray") -> "np.ndarray":
    """Close small gaps and fill every enclosed hole so the mask is a solid
    silhouette.

    SAM 2 tends to exclude dark surface details (here: the candy decorations on
    the cookie) as holes INSIDE the object. Downstream that is poison:
    TextureMesh treats hole pixels as ignore-label in every source view, blacks
    them out, and seam-leveling then smears that black across the whole atlas —
    the model renders near-black despite a correct-looking mask overlay. The
    silhouette must be solid; interior appearance is texture's job, not the
    mask's.
    """
    import cv2
    k = max(3, int(round(min(m.shape) * 0.004)) | 1)
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    # Flood-fill the background from the border; anything unreached is an
    # enclosed hole and becomes foreground.
    h, w = closed.shape
    ff = closed.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)
    holes = (ff == 0)
    out = closed.copy()
    out[holes] = 255
    return out


def main() -> None:
    ap = base_argparser("Stage 2: COLMAP sparse, masked vs unmasked")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"],
                    default="exhaustive")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--variants", default="unmasked,masked",
                    help="comma list; the brief asks for both")
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s2_colmap") and not args.force:
        log("[s2_colmap] already done (use --force to redo)")
        return

    global USE_GPU
    from common import device
    USE_GPU = "1" if device() == "cuda" else "0"

    with Stage(p, "s2_colmap") as st:
        have_masks = p.masks.exists() and any(p.masks.glob("*.png"))
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        if not have_masks:
            wanted = [v for v in wanted if v != "masked"]
            st.notes["masked_skipped"] = "no masks present — s1_masks did not run"
            log("  ! no masks found; running unmasked only")

        results = []
        for v in wanted:
            log(f"  --- variant: {v} ---")
            try:
                results.append(sparse(p, v, args.matcher, args.max_features))
            except Exception as e:
                results.append({"variant": v, "registered": 0, "error": str(e),
                                "seconds": 0})
                log(f"  ! variant {v} failed: {e}")

        ok = [r for r in results if r.get("registered", 0) > 0]
        if not ok:
            raise RuntimeError("no COLMAP variant registered any image")

        # More registered images wins; ties broken by lower reprojection error.
        def key(r):
            return (r["registered"], -(r.get("mean_reproj_err") or 99))
        winner = max(ok, key=key)

        st.notes["comparison"] = results
        st.notes["winner"] = winner["variant"]
        st.notes["winner_reason"] = (
            f"{winner['registered']}/{winner.get('n_images')} images registered, "
            f"mean reprojection error {winner.get('mean_reproj_err')} px"
            + (f"; other variant registered "
               f"{[r['registered'] for r in results if r is not winner]}"
               if len(results) > 1 else ""))
        st.notes["matcher"] = args.matcher

        # --- undistort the winner, and the masks alongside it, for OpenMVS
        model = Path(winner["model_dir"])
        und = undistort(p, model, p.resized, p.work / "undistorted")
        st.notes["undistorted"] = str(und)

        if have_masks:
            um = undistort(p, model, masks_as_images(p), p.work / "undistorted_masks")
            n = build_mvs_masks(p, um, p.work / "mvs_masks")
            st.notes["mvs_masks"] = n
            log(f"  wrote {n} undistorted OpenMVS masks")

        (p.work / "colmap_summary.json").write_text(
            json.dumps({"results": results, "winner": winner}, indent=2),
            encoding="utf-8")
        log(f"  winner: {winner['variant']} — {st.notes['winner_reason']}")


if __name__ == "__main__":
    main()
