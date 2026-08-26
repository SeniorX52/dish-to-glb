"""Stage 3B (path B) — RealityScan CLI, for comparison against path A.

On this machine this stage does not run, and that is a finding rather than a
failure. Epic's own hardware requirements page says, verbatim:

    "While a compatible NVIDIA GPU is recommended for optimal performance, the
     application can still run and perform image registration without one;
     however, you won't be able to create models or textures."
    -- https://dev.epicgames.com/documentation/realityscan/hardware-and-software-requirements

The two steps path B exists to produce -- -calculateNormalModel and
-calculateTexture -- are exactly the steps gated on a supported GPU. The
documented minimum is "CUDA Compute Capability 3.5 or higher"; RealityScan 2.2
(June 2026) broadened support to AMD RDNA 3/3.5/4 and Ryzen AI Max APUs, but
Intel GPUs appear nowhere in the supported list.

So the stage detects the executable and the GPU, records a structured skip with
that citation, and leaves the full command sequence below so the same repo
produces a real path-B comparison unchanged on the RTX 5080.

Licensing, for the client-facing cost note: RealityScan is free for artists and
studios under $1M/year gross revenue, $1,250/seat/year above that. The old
credit-based Pay-Per-Input model was discontinued in April 2024, so there is no
per-image recurring cost to model.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Paths, Stage, base_argparser, device, log, run, skip_note, stage_done

CITATION = ("https://dev.epicgames.com/documentation/realityscan/"
            "hardware-and-software-requirements")

CANDIDATES = [
    r"C:\Program Files\Epic Games\RealityScan\RealityScan.exe",
    r"C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe",
    r"C:\Program Files\Epic Games\RealityCapture\RealityCapture.exe",
]


def find_exe() -> str | None:
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    from common import which
    return which("RealityScan", "RealityCapture")


def build_commands(p: Paths, exe: str, out_obj: Path) -> list[str]:
    """The literal path-B invocation. Flags taken from Epic's command-line docs
    (https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm).

    Masks: RealityScan looks for <imagename>.mask.png alongside the image, so
    s2_colmap's undistorted masks are re-pointed at the ORIGINAL images here.
    """
    return [
        exe, "-headless",
        "-addFolder", str(p.images),
        "-align",
        "-setReconstructionRegionAuto",
        "-selectMaximalComponent",
        "-calculateNormalModel",
        "-calculateTexture",
        "-exportSelectedModel", str(out_obj),
        "-quit",
    ]


def main() -> None:
    args = base_argparser("Stage 3B: RealityScan CLI (comparison path)").parse_args()
    p = Paths(args.dataset).ensure()
    if stage_done(p, "s3b_realityscan") and not args.force:
        log("[s3b_realityscan] already done")
        return

    exe = find_exe()
    dev = device()

    if platform.system() != "Windows":
        skip_note(p, "s3b_realityscan",
                  "RealityScan is Windows-only; this host is " + platform.system())
        return

    if exe is None:
        skip_note(p, "s3b_realityscan",
                  "RealityScan/RealityCapture is not installed on this machine. It is "
                  "also unusable here: Epic requires a GPU with CUDA Compute Capability "
                  "3.5+ (or AMD RDNA 3/3.5/4 since 2.2) to create models or textures, "
                  "and this box has Intel integrated graphics only. Install + rerun on "
                  f"the CUDA machine to produce the comparison. Citation: {CITATION}")
        return

    if dev != "cuda":
        # Installed but the GPU gate applies: alignment would succeed and both
        # -calculateNormalModel and -calculateTexture would produce nothing.
        cmd = build_commands(p, exe, p.out / "realityscan_model.obj")
        skip_note(p, "s3b_realityscan",
                  "RealityScan is installed but no CUDA/supported GPU is present. Epic: "
                  "\"you won't be able to create models or textures\" without one, so "
                  "-calculateNormalModel and -calculateTexture would no-op. Not run, to "
                  "avoid reporting a fake comparison. Command that would run: "
                  + " ".join(cmd) + f" | Citation: {CITATION}")
        return

    out_obj = p.out / "realityscan_model.obj"
    with Stage(p, "s3b_realityscan") as st:
        cmd = build_commands(p, exe, out_obj)
        st.notes["command"] = " ".join(cmd)
        run(cmd, log_name="realityscan", check=True)
        st.notes["output"] = str(out_obj)
        st.notes["note"] = "RealityScan headless align + model + texture"
        log(f"  path B produced {out_obj}")


if __name__ == "__main__":
    main()
