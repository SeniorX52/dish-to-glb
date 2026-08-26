"""Stage 5B — render turntable screenshots of the final GLB for RESULTS.md.

Separate from s5_export so render time does not pollute the export timing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Paths, Stage, base_argparser, log, run, skip_note, stage_done

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_export import blender_exe  # noqa: E402


def main() -> None:
    ap = base_argparser("Stage 5B: render screenshots")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--res", type=int, default=720)
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s5b_render") and not args.force:
        log("[s5b_render] already done (use --force to redo)")
        return

    glb = p.out / "model.glb"
    if not glb.exists():
        skip_note(p, "s5b_render", f"{glb} does not exist — s5_export did not produce a GLB")
        return
    blender = blender_exe()
    if not blender:
        skip_note(p, "s5b_render", "Blender not found; cannot render screenshots")
        return

    with Stage(p, "s5b_render") as st:
        r = run([blender, "--background", "--factory-startup", "--python-exit-code", "1",
                 "--python", ROOT / "pipeline" / "blender_render.py",
                 "--", str(glb), str(p.out), str(args.views), str(args.res)],
                log_name="blender_render", check=False)
        m = re.search(r"BLENDER_RENDER_JSON (\{.*\})", (r.stdout or "") + (r.stderr or ""))
        if m:
            st.notes.update(json.loads(m.group(1)))
        made = sorted(p.out.glob("render_*.png"))
        st.notes["renders_written"] = [f.name for f in made]
        if not made:
            raise RuntimeError("Blender produced no renders — see logs/blender_render.log")
        log(f"  {len(made)} renders -> {p.out}")


if __name__ == "__main__":
    main()
