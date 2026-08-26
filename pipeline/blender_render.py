"""Runs INSIDE Blender. Renders N turntable views of a GLB to PNG.

    blender --background --factory-startup --python-exit-code 1 \
            --python pipeline/blender_render.py -- model.glb outdir 3 640

Cycles on CPU, deliberately: EEVEE needs a GL context that a --background
Windows session does not reliably have. Sample count is kept low because these
are documentation screenshots, not beauty renders.
"""
import json
import math
import sys

import bpy
from mathutils import Vector


def scene_bounds():
    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    found = False
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        found = True
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mins[i] = min(mins[i], w[i])
                maxs[i] = max(maxs[i], w[i])
    if not found:
        raise SystemExit("no mesh in scene")
    return mins, maxs


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    glb, outdir = argv[0], argv[1]
    n = int(argv[2]) if len(argv) > 2 else 3
    res = int(argv[3]) if len(argv) > 3 else 640

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)

    mins, maxs = scene_bounds()
    centre = (mins + maxs) / 2
    radius = max((maxs - mins).length / 2, 1e-6)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.resolution_x = res
    sc.render.resolution_y = int(res * 0.75)
    sc.render.film_transparent = False

    # Neutral studio-ish world so the texture is readable.
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.56, 0.58, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.6

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    key = bpy.data.objects.new("key", bpy.data.lights.new("key", type="AREA"))
    key.data.energy = 400 * radius * radius
    key.data.size = radius * 3
    sc.collection.objects.link(key)
    key.location = centre + Vector((radius * 2, -radius * 2, radius * 3))
    key.rotation_euler = (math.radians(35), 0, math.radians(35))

    written = []
    dist = radius * 3.1
    for i in range(n):
        az = math.radians(35 + i * (360.0 / max(n, 1)))
        el = math.radians(22)
        cam.location = centre + Vector((
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el)))
        d = centre - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        path = f"{outdir}/render_{i + 1}.png"
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)

    print("BLENDER_RENDER_JSON " + json.dumps(
        {"renders": written, "radius": radius,
         "centre": [round(c, 4) for c in centre]}))


if __name__ == "__main__":
    main()
