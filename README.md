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
| [SETUP.md](SETUP.md) | exact environment reproduction, every version pinned, every landmine documented |
| [ARCHITECTURE.md](ARCHITECTURE.md) | production design: phone vs backend split, cost model, licensing |

## The one finding worth reading first

Background isolation is not just a quality feature — it is the main per-scan cost
control. Masking during COLMAP *alignment* starved registration (60/85 images vs
83/85 unmasked), but masking during *densification* confined depth estimation to
~13% of each frame's pixels. Align unmasked, densify masked.

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
