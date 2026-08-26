# Production architecture recommendation

> Scope note: `RESULTS.md` contains **only measured numbers**. This file is where
> forward-looking **estimates** live, and every estimate is labelled as one. The
> two are kept separate on purpose — mixing them is how photogrammetry proposals
> end up promising throughput nobody ever benchmarked.

---

## 1. Recommended stack

**SAM 2.1 (video propagation) → COLMAP → OpenMVS → Draco/WebP GLB → `<model-viewer>`**

Why this and not the alternatives:

| Option | Verdict |
|---|---|
| **SAM 2.1 + COLMAP + OpenMVS** | **Chosen.** Fully open, permissively licensed, no per-scan fee, runs on any CUDA box, and every stage is scriptable and headless. The only stack here with *zero* recurring cost. |
| SAM 2.1 + RealityScan | Best-in-class alignment and speed, and genuinely worth using if the volume justifies a seat. But it is Windows-only, needs a supported GPU, and adds a per-seat licence. Keep as a quality benchmark, not the production spine. |
| KIRI Engine API | Fastest path to a result and removes all GPU ops burden, but it is a per-scan metered dependency on a third party, with your asset pipeline and margin structure hostage to their pricing. Reasonable as a v0 stopgap; poor as a foundation. |
| Gaussian splatting / NeRF | Beautiful for whole scenes and view-dependent effects. Wrong shape for this product: restaurant menus need a **watertight, textured, decimated mesh** that drops into `<model-viewer>` and AR Quick Look. Splats do not give you that without a meshing step that throws away their advantage. |

The decisive property for MANAH is that a dish is a **small, isolated, matte-to-glossy object on a cluttered surface**. That makes segmentation quality — not reconstruction cleverness — the dominant term.

---

## 2. Keeping the table out of the model

This is the part the brief is right to single out, and it has a specific answer:
**mask at depth-map estimation, not after meshing.**

The order matters more than the tool:

1. **One click, then propagate.** Treat the image set as a video and run SAM 2.1's
   video predictor: a single seed on frame 0, propagated across every frame.
   Per-image segmentation with a fresh prompt drifts, and neighbouring views end
   up disagreeing about where the object ends — which is precisely what injects
   table geometry. Implemented in `pipeline/s1_masks.py`.
2. **Align on the full frame, mask from densification onward.** Masking during
   feature extraction can starve alignment on a low-texture object. The pipeline
   does not assume this — it builds *both* sparse models and reports registered
   image count and mean reprojection error for each (`pipeline/s2_colmap.py`),
   then uses the winner. See the comparison table in `RESULTS.md`.
3. **Undistort the masks alongside the images.** After `colmap image_undistorter`
   the images no longer match the original mask geometry. The masks are run
   through the *same* undistortion so they stay pixel-aligned.
4. **`--ignore-mask-label 0` during `DensifyPointCloud`.** Background pixels never
   become depth samples, so the table is never in the point cloud to begin with.
5. **Guarded RANSAC plane cut as a backstop only** (`pipeline/s4_cleanup.py`). It
   refuses to fire when the dominant plane holds too few vertices — otherwise on a
   correctly-masked scan it would happily eat the flat underside of the dish.

Layers 1–4 do the real work. Layer 5 exists because capture conditions in a
restaurant will occasionally be worse than the lab.

### Two failure modes to design for
- **Glossy plating and glassware.** Specular highlights move with the camera and
  break both matching and depth. Mitigation: diffuse lighting at capture, and a
  polarising filter if the operator will tolerate one.
- **Near-featureless surfaces** (a flat sauce, a white plate). Mitigation is
  capture-side: keep some textured context in frame for alignment, then mask it
  out for densification — which is exactly why step 2 aligns unmasked.

---

## 3. What runs where

**On the phone / in the browser — capture and QA only. No reconstruction.**
- Guided multi-view capture with live coverage feedback (orbit count, angular gaps).
- Enforce fixed exposure/focus, reject blurry frames on-device using a cheap
  Laplacian-variance check before upload. Rejecting a bad frame at capture is
  ~1000× cheaper than discovering it after a 20-minute reconstruction.
- **One tap on the dish** to seed segmentation. This is the single highest-value
  piece of human input in the whole system: it costs the operator one second and
  removes the hardest ambiguity from the pipeline.
- Upload originals to object storage. Do **not** segment on-device — SAM 2.1
  hiera-large is not a phone workload, and you want the masks reproducible.

**On a GPU backend — everything else.**
- A queue (SQS/Pub-Sub) in front of stateless GPU workers, one scan per worker.
- The pipeline in this repo, unchanged, driven by `run.py --dataset <scan_id>`.
- Artefacts to object storage; CDN in front of the GLB/USDZ.
- The stage checkpointing in `work/<ds>/.state/` already gives you crash-resume
  for free: a worker that dies mid-scan resumes at the last completed stage.

**Serving.** `<model-viewer>` with a Draco-compressed GLB plus `ios-src` USDZ for
iOS Quick Look. Draco specifically, not meshopt — model-viewer auto-fetches a
Draco decoder with zero client config, whereas meshopt silently fails to load
unless you configure a decoder URL first. That default trips people constantly.

---

## 4. Cost and throughput

**Measured on this run:** see the stage table in `RESULTS.md`. Those numbers come
from a **15 W laptop CPU with no CUDA at all**, so they represent a worst case,
not a production figure.

**Estimated for production — these are estimates, not measurements:**

The dominant GPU-accelerable stages are SAM 2.1 propagation and OpenMVS
`DensifyPointCloud`. On a modern CUDA GPU both typically run an order of
magnitude faster than the CPU path measured here; COLMAP feature matching also
moves from CPU SIFT to GPU SIFT. A realistic target on a single mid-range cloud
GPU instance is **single-digit minutes per dish** for an 80–120 image capture at
this working resolution.

**Before quoting a client a per-scan price, run this repo once on the target GPU
instance and read the number out of `benchmark.json`.** That is a ~30-minute task
and it converts every figure above from an estimate into a measurement. I have
deliberately not invented a dollars-per-scan number here.

The cost model itself is simple and worth stating explicitly:

```
cost_per_scan ≈ (measured wall-clock hours on target instance) × (instance $/hr)
              + storage(originals + artefacts)
              + CDN egress × expected views
```

Storage will quietly dominate at scale: 100 originals at ~2 MB each is ~200 MB
per dish, against a ~2–3 MB delivered GLB. Plan a lifecycle policy that moves
originals to cold storage once a scan is approved — you need them only for
re-processing.

### Licensing and recurring cost

| Component | Licence | Recurring cost |
|---|---|---|
| SAM 2.1 | Apache 2.0 (model weights + code) | none |
| COLMAP | BSD | none |
| OpenMVS | AGPL-3.0 | none, **but see below** |
| Open3D | MIT | none |
| Blender | GPL-3.0 (as a separate CLI process) | none |
| glTF-Transform, model-viewer | MIT / Apache 2.0 | none |
| RealityScan (optional) | free under $1M/yr revenue; $1,250/seat/yr above | per seat, not per scan — PPI discontinued April 2024 |

> **AGPL flag — the one licensing item that needs a decision.** OpenMVS is
> AGPL-3.0. Invoking it as a separate binary in a backend batch job is the normal
> pattern, but AGPL's network clause is exactly the kind of thing a future
> acquirer's counsel will ask about. Two clean options: (a) keep OpenMVS as an
> arms-length subprocess and take legal advice, or (b) swap the densify/mesh
> stage for a permissively-licensed alternative. This is a business decision, not
> a technical one, and it should be made deliberately rather than by default.

---

## 5. Honest limitations of this PoC

- Texturing samples the **2000 px working images**, not the 4624 px originals.
  Using the originals requires rescaling the COLMAP intrinsics by the ingest
  factor and re-running undistortion at full resolution — straightforward, but
  not done here. At a 2048² atlas the visible difference is small.
- No colour calibration. Restaurant lighting varies wildly; a colour chart in the
  first frame and a per-scan white balance would be a cheap, large quality win.
- No automatic scale. The mesh is in arbitrary COLMAP units. For a menu that is
  mostly fine, but a known-size marker in frame gives real-world dimensions
  nearly free.
- Single object per scan. Multiple dishes need either separate captures or
  multi-object SAM prompts and per-object mesh splitting.
