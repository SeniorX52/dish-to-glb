"""Stage 0 — ingest.

Reads EXIF, writes a downscaled working copy, and lays out the numbered JPEG
frame directory that SAM 2's video predictor requires. Originals are never
touched; texturing reads them back later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ExifTags

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Paths, Stage, base_argparser, list_images, log, stage_done

MAX_DIM = 2000
TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _exif(img: Image.Image) -> dict:
    out = {}
    try:
        ex = img.getexif()
        if not ex:
            return out
        for name in ("Make", "Model", "FocalLength", "FocalLengthIn35mmFilm",
                     "ExposureTime", "ISOSpeedRatings", "LensModel"):
            tag = TAGS.get(name)
            if tag is not None and tag in ex:
                v = ex[tag]
                out[name] = float(v) if hasattr(v, "numerator") else str(v)
    except Exception:
        pass
    return out


def subsample(srcs: list[Path], keep: int) -> list[Path]:
    """Evenly spaced subset, preserving capture order so the sequence stays
    continuous for SAM 2's video propagation. Evenly spaced beats every-Nth
    when the count does not divide cleanly."""
    if keep <= 0 or keep >= len(srcs):
        return srcs
    import numpy as np
    idx = np.linspace(0, len(srcs) - 1, keep).round().astype(int)
    return [srcs[i] for i in sorted(set(idx.tolist()))]


def ingest(p: Paths, max_dim: int = MAX_DIM, max_images: int = 0) -> dict:
    srcs = list_images(p.images)
    if not srcs:
        raise RuntimeError(f"no images found in {p.images}")
    n_available = len(srcs)
    srcs = subsample(srcs, max_images)

    p.resized.mkdir(parents=True, exist_ok=True)
    p.frames.mkdir(parents=True, exist_ok=True)

    index, cameras, sizes, factors = [], {}, set(), []

    for i, src in enumerate(srcs):
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            sizes.add(f"{w}x{h}")
            if i == 0 or len(cameras) == 0:
                pass
            ex = _exif(im)
            cam = f"{ex.get('Make', '?')} {ex.get('Model', '?')}".strip()
            cameras[cam] = cameras.get(cam, 0) + 1
            if i == 0:
                first_exif = ex

            scale = min(1.0, max_dim / max(w, h))
            factors.append(round(scale, 4))
            if scale < 1.0:
                im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

            # Working copy keeps the original stem so COLMAP/OpenMVS mask naming lines up.
            dst = p.resized / (src.stem + ".jpg")
            im.save(dst, "JPEG", quality=95)
            # SAM 2's video predictor wants zero-padded numeric frame names.
            im.save(p.frames / f"{i:05d}.jpg", "JPEG", quality=95)

        index.append({"i": i, "orig": src.name, "work": dst.name,
                      "frame": f"{i:05d}.jpg", "orig_size": [w, h]})

    meta = {
        "n_images": len(srcs),
        "n_images_available": n_available,
        "subsampled": len(srcs) < n_available,
        "source_resolutions": sorted(sizes),
        "cameras": cameras,
        "first_image_exif": first_exif,
        "max_dim": max_dim,
        "resize_factor_min": min(factors),
        "resize_factor_max": max(factors),
        "working_dir": str(p.resized),
        "frames_dir": str(p.frames),
    }
    (p.work / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (p.work / "ingest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = base_argparser("Stage 0: ingest + downscale")
    ap.add_argument("--max-images", type=int, default=0,
                    help="evenly subsample to this many images (0 = use all)")
    ap.add_argument("--max-dim", type=int, default=MAX_DIM)
    args = ap.parse_args()
    p = Paths(args.dataset).ensure()
    if stage_done(p, "s0_ingest") and not args.force:
        log("[s0_ingest] already done (use --force to redo)")
        return
    with Stage(p, "s0_ingest") as st:
        meta = ingest(p, args.max_dim, args.max_images)
        st.notes.update(meta)
        log(f"  {meta['n_images']} images (of {meta['n_images_available']} available), "
            f"{meta['source_resolutions']}, cameras={meta['cameras']}, "
            f"scale {meta['resize_factor_min']}-{meta['resize_factor_max']}")


if __name__ == "__main__":
    main()
