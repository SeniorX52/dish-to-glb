"""Stage 1 — subject isolation with SAM 2.1.

The image set is treated as a video: one prompt on the first frame, then
propagation across every frame. Per-image prediction with a fresh prompt drifts
and produces masks that disagree between neighbouring views, which is exactly
what poisons a reconstruction.

Seeding has two modes:

  * `--click` opens a one-shot window to click the object; the clicks are cached
    in work/<ds>/prompt.json so every rerun is non-interactive. This mirrors the
    real product flow, where the operator taps the dish once.
  * Default (headless) computes the frame-0 mask automatically via
    auto_seed_mask() and seeds the tracker with that MASK.

Seeding with a mask rather than a point is deliberate. A centre point is a bad
prior for a plated dish: on the test capture it selected the icing decoration
*on top of* the cookie, and the tracker then followed that detail faithfully
until it rotated out of view, collapsing mask coverage from 5% to 0.02%. A mask
seed starts the tracker from something that has already been checked, and
outputs/<ds>/seed_check.jpg is written so it can be checked again by eye.

API verified against the installed sam2 source (init_state / add_new_mask /
add_new_points_or_box / propagate_in_video), not from memory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, Paths, Stage, base_argparser, device, log, stage_done

REPO = Path(__file__).resolve().parent.parent / "tools" / "sam2repo"
CKPT_DIR = REPO / "checkpoints"
MODELS = {
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
}


# ------------------------------------------------------------------ prompting
def auto_seed_mask(frame: Path, cfg: str, ckpt: Path, dev: str):
    """Produce the frame-0 mask automatically, then hand it to the video
    predictor as a MASK rather than as points.

    Two stages, because one stage does not work:
      1. A wide central box. A box says "segment the thing inside this", which
         reliably finds the dish body. Centre POINTS do not -- on the test dish
         a centre point selected the icing decoration on top of the cookie and
         the tracker then faithfully followed that detail until it rotated out
         of view.
      2. Tighten the box to the bounding box of what stage 1 found, pad it, and
         re-prompt. Stage 1 tends to clip attached parts (the cookie's head);
         the padded re-prompt recovers them without reaching out to the table,
         which a wider box or a scattered point grid both do.

    Returns (bool mask, info dict).
    """
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img = np.array(Image.open(frame).convert("RGB"))
    h, w = img.shape[:2]
    model = build_sam2(cfg, str(ckpt), device=dev)
    pred = SAM2ImagePredictor(model)
    pred.set_image(img)

    coarse = np.array([w * 0.06, h * 0.04, w * 0.94, h * 0.96])
    with torch.inference_mode():
        masks, scores, _ = pred.predict(box=coarse, multimask_output=True)
    base = masks[int(np.argmax(scores))] > 0
    if not base.any():
        raise RuntimeError("SAM 2 found nothing inside the coarse box")

    ys, xs = np.where(base)
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    pad = 0.28
    tight = np.array([max(0, xs.min() - bw * pad), max(0, ys.min() - bh * pad),
                      min(w, xs.max() + bw * pad), min(h, ys.max() + bh * pad)])
    centre = np.array([[(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2]])
    with torch.inference_mode():
        masks2, scores2, _ = pred.predict(box=tight, point_coords=centre,
                                          point_labels=np.array([1]),
                                          multimask_output=True)
    refined = masks2[int(np.argmax(scores2))] > 0

    # Guard: if refining more than doubled the area it has probably escaped onto
    # the table, so keep the conservative stage-1 mask instead.
    info = {"coarse_coverage": round(float(base.mean()), 4),
            "refined_coverage": round(float(refined.mean()), 4),
            "refined_score": round(float(scores2[int(np.argmax(scores2))]), 4)}
    if refined.mean() > max(2.0 * base.mean(), 0.45):
        info["refine_rejected"] = ("refined mask grew beyond 2x the coarse mask "
                                   "(or past 45% of frame) — likely leaked onto "
                                   "the background; kept the coarse mask")
        del pred, model
        return base, info
    del pred, model
    return refined, info


def click_prompt(frame: Path) -> dict:
    """Tiny interactive helper: left click = object, right click = background."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    img = Image.open(frame).convert("RGB")
    pos: list[list[float]] = []
    neg: list[list[float]] = []
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img)
    ax.set_title("LEFT-click the object (1+ points)  |  RIGHT-click the table\n"
                 "close the window when done")
    ax.axis("off")

    def on_click(ev):
        if ev.xdata is None:
            return
        if ev.button == 1:
            pos.append([float(ev.xdata), float(ev.ydata)])
            ax.plot(ev.xdata, ev.ydata, "o", color="lime", markersize=9)
        elif ev.button == 3:
            neg.append([float(ev.xdata), float(ev.ydata)])
            ax.plot(ev.xdata, ev.ydata, "x", color="red", markersize=11)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    if not pos:
        raise RuntimeError("no positive click recorded")
    return {"auto": False, "points": pos, "labels": [1] * len(pos), "neg_points": neg}


def load_prompt(p: Paths, frames: list[Path], interactive: bool) -> dict:
    """Returns {'mode': 'click'|'mask'}; a click prompt also carries its points."""
    f = p.work / "prompt.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        log(f"  using cached prompt (mode={d.get('mode')})")
        return d
    if interactive:
        d = click_prompt(frames[0])
        d["mode"] = "click"
        f.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return d
    return {"mode": "mask"}


def save_seed_overlay(frame: Path, mask, out: Path) -> None:
    """Auditable proof of what the seed actually selected."""
    import numpy as np
    img = Image.open(frame).convert("RGB")
    a = np.array(img).astype(np.float32)
    m = np.asarray(mask, dtype=bool)
    a[~m] *= 0.30
    a[m] = a[m] * 0.6 + np.array([60, 255, 90]) * 0.4
    t = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    t.thumbnail((900, 620))
    out.parent.mkdir(parents=True, exist_ok=True)
    t.save(out, "JPEG", quality=90)


# ------------------------------------------------------------------ contact sheet
def contact_sheet(p: Paths, index: list[dict], out: Path, cols: int = 4, rows: int = 3) -> None:
    n = cols * rows
    if not index:
        return
    picks = [index[i] for i in np.linspace(0, len(index) - 1, min(n, len(index))).astype(int)]
    tw, th = 420, 315
    sheet = Image.new("RGB", (cols * tw, rows * th), (18, 18, 20))
    from PIL import ImageDraw
    for k, rec in enumerate(picks):
        img = Image.open(p.resized / rec["work"]).convert("RGB")
        mpath = p.masks / f"{Path(rec['work']).stem}.png"
        if mpath.exists():
            m = np.array(Image.open(mpath).convert("L").resize(img.size)) > 127
            a = np.array(img).astype(np.float32)
            a[~m] *= 0.28                      # dim whatever was cut away
            a[m] = a[m] * 0.82 + np.array([0, 90, 30]) * 0.18   # tint what survived
            img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        img.thumbnail((tw, th))
        x, y = (k % cols) * tw, (k // cols) * th
        sheet.paste(img, (x + (tw - img.width) // 2, y + (th - img.height) // 2))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), rec["work"], fill=(255, 220, 90))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=88)
    log(f"  contact sheet -> {out}")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = base_argparser("Stage 1: SAM 2.1 subject isolation")
    ap.add_argument("--model", choices=list(MODELS), default="large")
    ap.add_argument("--click", action="store_true",
                    help="open a window to click the object once; cached afterwards")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N frames")
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s1_masks") and not args.force:
        log("[s1_masks] already done (use --force to redo)")
        return

    index = json.loads((p.work / "index.json").read_text(encoding="utf-8"))
    frames_dir = p.frames
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames in {frames_dir} — run s0_ingest first")
    if args.limit:
        # init_state reads the whole directory, so a subset needs its own folder
        # with contiguous 00000.jpg.. names or the loader's int() sort breaks.
        import shutil
        sub = p.work / "frames_subset"
        shutil.rmtree(sub, ignore_errors=True)
        sub.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames[:args.limit]):
            shutil.copy(f, sub / f"{i:05d}.jpg")
        frames_dir = sub
        frames = sorted(sub.glob("*.jpg"))
        index = index[:args.limit]

    cfg, ckpt = MODELS[args.model]
    ckpt_path = CKPT_DIR / ckpt
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint missing: {ckpt_path}\n"
                           f"download from https://dl.fbaipublicfiles.com/"
                           f"segment_anything_2/092824/{ckpt}")

    with Stage(p, "s1_masks") as st:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        dev = device()
        torch.set_num_threads(os.cpu_count() or 8)
        log(f"  device={dev} model={args.model} frames={len(frames)}")

        prompt = load_prompt(p, frames, args.click)

        seed_mask = None
        if prompt.get("mode") == "mask":
            seed_mask, seed_info = auto_seed_mask(frames[0], cfg, ckpt_path, dev)
            st.notes["seed"] = seed_info
            save_seed_overlay(frames[0], seed_mask, p.out / "seed_check.jpg")
            log(f"  auto seed mask: {seed_info}")

        predictor = build_sam2_video_predictor(cfg, str(ckpt_path), device=dev)

        # CPU path stays in float32: bfloat16 autocast on CPU is slower here and
        # changes results. CUDA gets bfloat16, which is what SAM 2 ships with.
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" \
            else torch.inference_mode()

        p.masks.mkdir(parents=True, exist_ok=True)
        coverage = {}
        with ctx:
            # Offloading only helps when the compute device is CUDA and host RAM
            # is the cheaper place to park frames/state. On a CPU device the data
            # is already in host RAM, so offloading buys nothing and costs a lot:
            # measured 38 s/frame with offload on vs 14 s/frame with it off.
            offload = (dev == "cuda")
            state = predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=offload,
                offload_state_to_cpu=offload,
            )
            st.notes["offload"] = offload
            if seed_mask is not None:
                # Seeding with the verified frame-0 mask beats seeding with a
                # guessed point: the tracker starts from exactly what we checked.
                predictor.add_new_mask(inference_state=state, frame_idx=0,
                                       obj_id=1, mask=torch.as_tensor(seed_mask))
            else:
                pts = np.array(prompt["points"] + prompt.get("neg_points", []),
                               dtype=np.float32)
                lbl = np.array(prompt["labels"] + [0] * len(prompt.get("neg_points", [])),
                               dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=pts, labels=lbl)

            written = 0
            for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                if fidx >= len(index):
                    continue
                m = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                # Solid silhouette: SAM 2 excludes dark surface details (candy,
                # sauce, char marks) as interior holes, and those holes black out
                # the texture atlas downstream. See s2_colmap.solidify_mask.
                from s2_colmap import solidify_mask
                m = solidify_mask(m)
                rec = index[fidx]
                stem = Path(rec["work"]).stem
                Image.fromarray(m).save(p.masks / f"{stem}.png")
                coverage[stem] = round(float((m > 0).mean()), 5)
                written += 1
                if written % 20 == 0:
                    log(f"    {written}/{len(frames)} masks")

        lost = [k for k, v in coverage.items() if v < 0.005]
        vals = sorted(coverage.values())
        st.notes.update(
            model=args.model, device=dev, n_masks=written,
            prompt_mode=prompt.get("mode"),
            coverage_min=vals[0] if vals else 0,
            coverage_median=vals[len(vals) // 2] if vals else 0,
            coverage_max=vals[-1] if vals else 0,
            frames_with_collapsed_mask=len(lost),
            seconds_per_frame=None,   # filled by Stage timing / n_masks in the report
        )
        if lost:
            st.notes["collapsed_frames"] = lost[:20]
            log(f"  ! {len(lost)} frames have <0.5% mask coverage — tracking may have drifted")
        (p.work / "mask_coverage.json").write_text(json.dumps(coverage, indent=1),
                                                   encoding="utf-8")
        contact_sheet(p, index, p.out / "mask_check.jpg")
        log(f"  {written} masks, coverage median {st.notes['coverage_median']:.3f}")


if __name__ == "__main__":
    main()
