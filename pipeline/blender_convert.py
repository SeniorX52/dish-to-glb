"""Runs INSIDE Blender, not in the venv:

    blender --background --factory-startup --python-exit-code 1 \
            --python pipeline/blender_convert.py -- in.obj out.glb out.usdz

Blender 4.0 deleted bpy.ops.import_scene.obj; on 5.x the OBJ importer is
bpy.ops.wm.obj_import. USDZ has no format argument at all -- the .usdz file
extension IS the format switch, and unlike the glTF exporter there is no
extension-correcting helper, so a wrong suffix silently yields plain USD.
"""
import json
import sys

import bpy


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("usage: -- <in.obj> <out.glb> [out.usdz]")
    obj_in, glb_out = argv[0], argv[1]
    usdz_out = argv[2] if len(argv) > 2 else None

    result = {"glb": None, "usdz": None, "usdz_error": None, "objects": 0,
              "blender": bpy.app.version_string}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=obj_in)          # 4.0+ name
    result["objects"] = len(bpy.context.scene.objects)
    if result["objects"] == 0:
        raise SystemExit(f"nothing imported from {obj_in}")

    tris = 0
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            tris += len(o.data.loop_triangles) or len(o.data.polygons)
    result["polygons"] = tris

    # Axis round-trip needs no manual rotation: wm.obj_import defaults to
    # forward NEGATIVE_Z / up Y, and the glTF exporter's export_yup default
    # converts Blender Z-up back to glTF Y-up.
    bpy.ops.export_scene.gltf(
        filepath=glb_out,
        export_format="GLB",
        export_materials="EXPORT",
        export_image_format="AUTO",
        use_selection=False,
        export_apply=True,
    )
    result["glb"] = glb_out

    if usdz_out:
        try:
            if not usdz_out.lower().endswith(".usdz"):
                raise ValueError("output path must end in .usdz or Blender "
                                 "silently writes plain USD")
            bpy.ops.wm.usd_export(
                filepath=usdz_out,
                export_materials=True,
                generate_preview_surface=True,   # required for iOS Quick Look shading
                export_uvmaps=True,
                export_normals=True,
                usdz_downscale_size="2048",      # enum values are STRINGS
            )
            result["usdz"] = usdz_out
        except Exception as e:                   # noqa: BLE001 - report, do not fake
            result["usdz_error"] = f"{type(e).__name__}: {e}"

    print("BLENDER_RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
