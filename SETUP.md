# SETUP — reproducing this environment from scratch

Windows 11 x64, native. **No WSL, no CUDA, no conda.** Every version below is the
version that actually ran, read back off the installed artefact rather than
copied from a tutorial.

---

## 0. The hardware this was built on (and what that changed)

The brief specified an RTX 5080 (sm_120), Core Ultra 9 285K, 48 GB RAM, WSL2.
The machine available was materially different, and it is worth stating plainly
because it shapes every number in `RESULTS.md`:

| Brief | Actually used |
|---|---|
| RTX 5080, 16 GB, sm_120 | **Intel(R) Graphics — integrated, no NVIDIA GPU** |
| Core Ultra 9 285K | Core Ultra 7 155U (12 threads, 15 W laptop class) |
| 48 GB RAM | 31.5 GB |
| WSL2 available | not installed, not used |
| CUDA 12.8 + cu128 torch | `torch 2.13.0+cpu` |

`torch.cuda.get_device_capability()` can never return `(12, 0)` here, so the
brief's environment gate cannot be satisfied. Rather than stop, the pipeline was
built CPU-viable **and** CUDA-autodetecting: `pipeline/common.py::device()` is
the single switch, and every stage honours it. The same `python run.py` command
runs unchanged on the RTX 5080 and will pick up CUDA automatically.

What this costs, concretely: path B (RealityScan) cannot run at all here, and
COLMAP's `patch_match_stereo` is CUDA-only so dense reconstruction goes through
OpenMVS's CPU PatchMatch instead. See "What does not work here" below.

---

## 1. Python

Python **3.14.4** was the only interpreter present and PyTorch publishes no
wheels for it. Install 3.12:

```powershell
winget install --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
py -0p     # confirm: -V:3.12  ...\Python\Python312\python.exe
```

Installed: **Python 3.12.10**.

```powershell
cd "<repo>"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

pip **26.2.1**.

### Python packages

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pillow psutil opencv-python tqdm trimesh matplotlib
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install open3d
```

Resolved to:

| package | version |
|---|---|
| torch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| numpy | 2.5.2 |
| open3d | 0.19.0 |
| opencv-python | 5.0.0.93 |
| pillow | 12.3.0 |
| psutil | 7.2.2 |
| trimesh | 5.0.0 |
| matplotlib | 3.11.1 |

> **On the RTX 5080 machine**, replace only the torch line — and **not with cu128**:
>
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
> python -c "import torch; print(torch.cuda.get_device_capability())"   # expect (12, 0)
> ```
>
> The brief says "cu128 wheels or later", and the *later* is doing real work
> there. Checked against the live PyTorch indexes on 2026-08-26, Windows cp312:
>
> | index | newest torch with a `win_amd64` cp312 wheel |
> |---|---|
> | `cu126` | 2.13.0 — but CUDA 12.6 predates Blackwell, so **no sm_120 kernels** |
> | `cu128` | **2.11.0** — frozen; 12.8/12.9 builds were dropped after torch 2.12 |
> | `cu129` | 2.9.0 |
> | `cu130` | **2.13.0** ← use this |
>
> Pinning `cu128` today silently downgrades you to torch 2.11. Confirm sm_120
> appears in `get_arch_list()` before trusting the install. Do **not** install any
> repo pinning torch ≤ 2.0, and do not use the original INRIA
> `diff-gaussian-rasterization` — use `gsplat` or a maintained fork if a splat
> path is attempted at all.

> Open3D 0.19.0 is the ceiling at Python 3.12 — there are no cp313/cp314 wheels.
> Do not move this venv past 3.12 without checking that first.

---

## 2. SAM 2.1

```powershell
cd tools
git clone --depth 1 https://github.com/facebookresearch/sam2.git sam2repo
cd sam2repo
# Bash, so the env vars apply to the pip subprocess:
SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 "<repo>/.venv/Scripts/python.exe" -m pip install -e .
```

Checkpoints (URLs taken from the repo's own `checkpoints/download_ckpts.sh`):

```bash
cd tools/sam2repo/checkpoints
curl -sSL -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt   # 898 MB
curl -sSL -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt   # 184 MB
```

Config names, confirmed against `tools/sam2repo/sam2/configs/sam2.1/`:
`configs/sam2.1/sam2.1_hiera_l.yaml`, `..._s.yaml`, `..._b+.yaml`, `..._t.yaml`.

**Expected warning, safe to ignore** — a CPU install has no compiled `_C`
extension:

```
UserWarning: cannot import name '_C' from 'sam2'
Skipping the post-processing step due to the error above.
```

Frame naming is not negotiable: `load_video_frames_from_jpg_images` sorts with
`int(os.path.splitext(name)[0])`, so frames must be `00000.jpg`, `00001.jpg`, …
`s0_ingest` writes exactly that into `work/<ds>/frames_jpg/`.

---

## 3. COLMAP

No winget package. Prebuilt Windows binaries come from GitHub releases, and
there is a **no-CUDA build**, which is the one to take here:

```powershell
cd tools
Invoke-WebRequest -Uri "https://github.com/colmap/colmap/releases/download/4.1.1/colmap-x64-windows-nocuda.zip" -OutFile colmap.zip
Expand-Archive colmap.zip .; Remove-Item colmap.zip
.\bin\colmap.exe help     # -> "COLMAP 4.1.1 (Commit a0d785f on 2026-07-17 without CUDA)"
```

**COLMAP 4.1.1 renamed the flags every tutorial still uses.** Read from the
binary's own `--help`, not from memory:

| what you want | 4.1.1 flag | the old name that no longer works |
|---|---|---|
| CPU SIFT extraction | `--FeatureExtraction.use_gpu 0` | `--SiftExtraction.use_gpu` |
| CPU SIFT matching | `--FeatureMatching.use_gpu 0` | `--SiftMatching.use_gpu` |
| feature cap | `--SiftExtraction.max_num_features` | *(unchanged)* |
| masks | `--ImageReader.mask_path <dir>` | *(unchanged)* |

`patch_match_stereo` is CUDA-only and is **not used** by this pipeline.

---

## 4. OpenMVS

```powershell
cd tools
Invoke-WebRequest -Uri "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip" -OutFile openmvs.zip
New-Item -ItemType Directory -Force openmvs | Out-Null
Expand-Archive openmvs.zip openmvs; Remove-Item openmvs.zip
```

Binaries land in `tools/openmvs/vc17/x64/Release/`. This is the **non-CUDA**
build and it runs fine on this CPU — its own log confirms the host:

```
OpenMVS x64 v2.4.0
CPU: Intel(R) Core(TM) Ultra 7 155U (14 cores)
SSE & AVX compatible CPU & OS detected
```

**OpenMVS writes all output to a `.log` file in the working directory, not to
stdout.** `DensifyPointCloud --help` prints nothing and exits 1; the help text
is in `DensifyPointCloud-<timestamp>.log`. That is where every flag in
`pipeline/s3_mvs.py` was verified.

**Known v2.4.0 bug — black texture patches.** v2.4.0's TextureMesh has a
seam-leveling sampler regression
([issue #1251](https://github.com/cdcseacave/openMVS/issues/1251)): a wrong
template parameter in `Types.inl` corrupts sampled pixels during seam leveling,
covering 25–35% of every texture patch in pure black; the model renders
near-black. Both the global AND local passes are affected (measured on this
dataset: 0.36 / 0.24 dark fraction vs 0.013 with both off). Fixed in develop
commit `eeedab7a` (2026-02-27) but **in no release yet**. `pipeline/s5_export.py`
passes `--global-seam-leveling 0 --local-seam-leveling 0` as the workaround; the
cost is unblended exposure seams between patches.

Two dead ends, so nobody repeats them: v2.3.0's only Windows binary links
`nvcuda.dll` (CUDA build) and cannot run without an NVIDIA driver; and
`--ignore-mask-label -2` reproducibly **segfaults** v2.4.0's TextureMesh, so
"mask handling off" must be expressed by omitting the flag instead.

---

## 5. Blender

```powershell
winget install --id BlenderFoundation.Blender --accept-source-agreements --accept-package-agreements --silent
```

Installed **Blender 5.2.1 LTS** (build date 2026-08-25) at
`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`. It is **not**
added to PATH; `pipeline/s5_export.py::blender_exe()` discovers it.

Two API facts that break older scripts:

* `bpy.ops.import_scene.obj` was **deleted in Blender 4.0**. The OBJ importer is
  now `bpy.ops.wm.obj_import`.
* `bpy.ops.wm.usd_export` has **no format argument**. The `.usdz` file extension
  *is* the format switch, and unlike the glTF exporter there is no
  extension-correcting helper — a wrong suffix silently writes plain USD.

---

## 6. Node tooling — and the one install that is actively broken

```powershell
cd tools\gltf
npm install          # package.json in this repo already contains the fix
```

**Do not run `npm install --global @gltf-transform/cli`.** On Windows,
`@gltf-transform/cli@4.4.2` pins `sharp ~0.34.5` while its transitive dependency
`ndarray-pixels@5.2.0` requires `sharp ^0.35.0`. The ranges are disjoint, so npm
installs *two* native libvips builds into one process, and every texture command
(`webp`, `resize`, and `optimize`'s texture step) dies with:

```
error: colourspace: parameter space not set
```

Geometry-only commands still work, which makes it easy to misdiagnose. The fix,
already in `tools/gltf/package.json`, is an `overrides` block forcing one sharp:

```json
{ "dependencies": { "@gltf-transform/cli": "4.4.2", "obj2gltf": "3.2.0" },
  "overrides": { "sharp": "0.35.4" } }
```

Verify: `npm ls sharp` must show a single `sharp@0.35.4 overridden` / `deduped`.

Versions: node **v24.15.0**, npm **11.16.0**, `@gltf-transform/cli` **4.4.2**,
sharp **0.35.4** / libvips **8.18.6**, obj2gltf **3.2.0**.

---

## 7. Dataset

```bash
mkdir -p data/gingerbread
curl -sSL -o Gingy.zip https://downloads.realityscan.com/Gingy.zip   # 330,030,750 bytes
# extract, then move the archive's Images/ folder to data/gingerbread/images/
```

No Epic account, no launcher, no EULA click-through — plain S3/CloudFront, HTTP 200.

Note Epic's own page says "158 images"; **the archive actually contains 171**,
all 4624×2604, EXIF `samsung SM-A715F` (Galaxy A71). The run reported in
`RESULTS.md` evenly subsamples these to 85 via `--max-images 85`.

Your own capture drops in with zero code changes:

```
data/own_dish/images/*.jpg
python run.py --dataset own_dish
```

---

## 8. Running it

```powershell
python run.py --dataset gingerbread                    # full chain
python run.py --dataset gingerbread --only s1_masks    # one stage
python run.py --dataset gingerbread --from s4_cleanup  # resume
python run.py --dataset gingerbread --force            # ignore checkpoints
```

Stages checkpoint into `work/<ds>/.state/`; a completed stage is skipped on
rerun unless `--force`. Stage scripts are individually runnable:
`python pipeline/s2_colmap.py --dataset gingerbread`.

Viewing (must be over HTTP — `fetch()` of `benchmark.json` is blocked on `file://`):

```powershell
python -m http.server 8000
# http://localhost:8000/viewer/?dataset=gingerbread
```

Publishing the viewer to GitHub Pages (**these are not run for you**):

```bash
git add viewer outputs/gingerbread/model.glb outputs/gingerbread/benchmark.json
git commit -m "MANAH PoC viewer"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
# then: repo Settings -> Pages -> Source: main / root
# viewer at https://<you>.github.io/<repo>/viewer/?dataset=gingerbread
```

---

## 9. What does not work on this machine

| thing | why | evidence |
|---|---|---|
| RealityScan / RealityCapture (path B) | needs a supported GPU to build models or textures | Epic, verbatim: *"While a compatible NVIDIA GPU is recommended for optimal performance, the application can still run and perform image registration without one; however, you won't be able to create models or textures."* Minimum is *"CUDA Compute Capability 3.5 or higher"*. 2.2 added AMD RDNA 3/3.5/4 only — no Intel. |
| `colmap patch_match_stereo` | CUDA-only | not used; OpenMVS CPU PatchMatch is used instead |
| Gaussian splatting / NeRF | needs CUDA | out of scope for this run |
| `gltf-transform` KTX2/ETC1S | needs the external KTX-Software `ktx` binary (not on PATH, not in winget) | WebP is used instead and needs no external tool |

RealityScan licensing, for the cost section of the proposal: free for artists and
studios under **$1M/year** gross revenue, **$1,250/seat/year** above that. The old
credit-based Pay-Per-Input model was **discontinued in April 2024**, so there is
no per-image recurring cost to model.
