# dish-to-glb

Phone photos of a dish on a table → clean, textured, web-ready 3D model.
One command, nine stages, every number measured.

![Reconstructed gingerbread cookie](outputs/gingerbread/render_3.png)

**Input:** 85 phone photos orbiting a decorated cookie on a wooden stump, outdoors
(Epic's RealityScan "Gingy" sample, Samsung Galaxy A71).
**Output:** a 0.76 MB Draco+WebP GLB — cookie only, no stump, no table, one
connected component, zero floaters. Total pipeline time **31.9 min on a CPU-only
laptop** (no CUDA involved anywhere).

## Pipeline

```
photos ─► s0 ingest ─► s1 SAM 2.1 isolation ─► s2 COLMAP sparse (masked vs unmasked, measured)
       ─► s3 OpenMVS densify+mesh (masked) ─► s4 cleanup ─► s5 texture + GLB/USDZ ─► s6 report
```

```bash
python run.py --dataset gingerbread     # full chain, checkpointed per stage
python run.py --dataset own_dish        # your photos in data/own_dish/images/ — zero code changes
```

| doc | contents |
|---|---|
| [RESULTS.md](RESULTS.md) | measured stage timings, masked-vs-unmasked alignment table, renders |
| [CALIBRATION_NOTE.md](CALIBRATION_NOTE.md) | printing and calipering the ChArUco board — read before capture |
| [SETUP.md](SETUP.md) | exact environment reproduction, every version pinned, every landmine documented |
| [ARCHITECTURE.md](ARCHITECTURE.md) | production design: phone vs backend split, cost model, licensing |

## The one finding worth reading first

Background isolation is not just a quality feature — it is the main per-scan cost
control. Masking during COLMAP *alignment* starved registration (60/85 images vs
83/85 unmasked), but masking during *densification* confined depth estimation to
~13% of each frame's pixels. Align unmasked, densify masked.

## Metric scale — the printed ChArUco board

```bash
python pipeline/make_charuco_pdf.py     # -> outputs/charuco/charuco_a4_7x6_25mm.pdf
```

A4, 7×6 grid, 25.00 mm squares, drawn as **vector** geometry so it prints at the
printer's own resolution rather than at a baked-in dpi. The generator rasterises
its own PDF, runs OpenCV's ChArUco detector over it and diffs it against
`CharucoBoard.generateImage()`; it refuses to report success unless every marker
decodes to the right id in the right square and the square spacing is within two
raster pixels of nominal.

**Printing it is not the last step.** Consumer printers rescale — "fit to page"
on an A4→Letter driver is a 3.4% error, and that lands directly on every
dimension the pipeline reports. The sheet carries two solid reference bars
(125.00 mm and 75.00 mm) to measure with calipers; the readings go in
`config/charuco.json` and *that* is what gets used. Until they do,
`effective_square_mm` is `None`, not `25.0`. See
[CALIBRATION_NOTE.md](CALIBRATION_NOTE.md).

## Viewer

```bash
python -m http.server 8000
# http://localhost:8000/viewer/?dataset=gingerbread
```

Static `<model-viewer>` page, no build step. Draco decoding needs zero client
config; the caption row is populated live from `benchmark.json`.

## Honest limits

- Texture seams are unblended: OpenMVS v2.4.0 ships a seam-leveling regression
  that blacks out texture patches ([#1251](https://github.com/cdcseacave/openMVS/issues/1251));
  this repo works around it and documents the cost.
- Edge-on views of thin objects are the weakest geometry.
- Glossy / saucy dishes are the open question — specular highlights move with the
  camera and break matching.
