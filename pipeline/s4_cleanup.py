"""Stage 4 — geometry cleanup: cut the residual table plane, drop floaters,
decimate to a web budget.

This runs on the UNTEXTURED mesh, before s5_export calls TextureMesh. That
ordering is deliberate and load-bearing: every CPU decimator available here
(Open3D simplify_quadric_decimation, trimesh, fast_simplification, COLMAP
mesh_simplifier) silently discards UVs. Decimating first and texturing last
means UVs are generated on the final geometry and never have to survive a
simplification pass.

The plane cut is a backstop for the UNMASKED path only, and it is off by default
when s3_mvs densified with masks. That is not caution for its own sake — it was
measured. On the test dish, with masking already doing its job, the dominant
RANSAC plane WAS the cookie's own flat face, and cutting it deleted 22.5% of the
object. When masks are used there is no table in the depth maps to remove, so
the correct number of planes to cut is zero. `--force-plane-cut` overrides.

When it does run (unmasked input), it still declines unless the plane holds at
least --plane-inlier-frac of the vertices and the cut would leave a non-trivial
mesh behind. Every decision, including declining, is recorded and surfaced in
RESULTS.md rather than silently swallowed.

Open3D 0.19 API signatures verified: segment_plane(distance_threshold, ransac_n,
num_iterations) -> (plane[4], inlier_indices); cluster_connected_triangles() ->
(triangle_cluster_ids, cluster_n_triangles, cluster_area);
simplify_quadric_decimation(target_number_of_triangles) -- an absolute count,
not a ratio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Paths, Stage, base_argparser, log, read_stages, stage_done


def _note_from_state(p: Paths, stage: str, key: str, default=None):
    """Read a note another stage recorded, so s4 can adapt to how s3 actually ran."""
    for s in read_stages(p):
        if s["name"] == stage:
            return s.get("notes", {}).get(key, default)
    return default


def load(path: Path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.triangles) == 0:
        raise RuntimeError(f"{path} has no triangles")
    return m


def counts(m) -> tuple[int, int]:
    return len(m.vertices), len(m.triangles)


def cut_plane(m, thresh_ratio: float, min_inlier_frac: float, notes: dict):
    """RANSAC the dominant plane; remove it and everything on its far side."""
    import open3d as o3d

    v = np.asarray(m.vertices)
    extent = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    dist = extent * thresh_ratio

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    plane, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3,
                                       num_iterations=2000)
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)
    nn = n / (np.linalg.norm(n) + 1e-12)
    frac = len(inliers) / max(len(v), 1)

    notes["plane_inlier_fraction"] = round(frac, 4)
    notes["plane_normal"] = [round(x, 4) for x in nn.tolist()]
    notes["plane_distance_threshold"] = round(dist, 6)

    if frac < min_inlier_frac:
        notes["plane_cut"] = (
            f"declined — dominant plane holds only {frac:.1%} of vertices "
            f"(< {min_inlier_frac:.0%}); masking appears to have removed the table "
            f"already, so cutting would risk flattening the object's underside")
        log(f"  plane cut declined ({frac:.1%} inliers)")
        return m, False

    # Which side is the object on? Use the centroid of the non-inlier vertices.
    signed = v @ nn + d / (np.linalg.norm(n) + 1e-12)
    mask_in = np.zeros(len(v), dtype=bool)
    mask_in[np.asarray(inliers)] = True
    obj_side = np.sign(np.mean(signed[~mask_in])) if (~mask_in).any() else 1.0
    if obj_side == 0:
        obj_side = 1.0

    # Keep only vertices comfortably on the object side of the plane.
    keep = (signed * obj_side) > dist
    removed = int((~keep).sum())
    if keep.sum() < 100:
        notes["plane_cut"] = ("declined — cut would leave fewer than 100 vertices; "
                              "the fitted plane was probably the object itself")
        log("  plane cut declined (would erase the object)")
        return m, False

    m.remove_vertices_by_mask(~keep)
    notes["plane_cut"] = (f"removed {removed} vertices "
                          f"({removed / len(v):.1%}) on/below the dominant plane")
    log(f"  plane cut: removed {removed} vertices ({removed / len(v):.1%})")
    return m, True


def keep_components(m, min_frac: float, notes: dict):
    """Keep the largest connected component plus anything within min_frac of it."""
    import open3d as o3d
    ids, n_tris, _area = m.cluster_connected_triangles()
    ids = np.asarray(ids)
    n_tris = np.asarray(n_tris)
    if len(n_tris) == 0:
        return m
    biggest = int(n_tris.max())
    keep_clusters = set(np.where(n_tris >= max(1, int(biggest * min_frac)))[0].tolist())
    drop = np.array([i not in keep_clusters for i in ids], dtype=bool)
    notes["components_total"] = int(len(n_tris))
    notes["components_kept"] = int(len(keep_clusters))
    notes["floater_triangles_removed"] = int(drop.sum())
    if drop.any():
        m.remove_triangles_by_mask(drop)
        m.remove_unreferenced_vertices()
    log(f"  components: {len(n_tris)} found, kept {len(keep_clusters)}, "
        f"dropped {int(drop.sum())} floater triangles")
    return m


def fill_holes(m, notes: dict):
    """Tensor-API hole fill; skipped cleanly if unavailable in this build."""
    import open3d as o3d
    try:
        v = np.asarray(m.vertices)
        extent = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
        t = o3d.t.geometry.TriangleMesh.from_legacy(m)
        filled = t.fill_holes(hole_size=extent * 0.02)
        out = filled.to_legacy()
        if len(out.triangles) >= len(m.triangles):
            notes["holes_filled_triangles"] = len(out.triangles) - len(m.triangles)
            return out
        notes["hole_fill"] = "skipped — fill_holes reduced triangle count"
    except Exception as e:
        notes["hole_fill"] = f"skipped — {type(e).__name__}: {e}"
    return m


def main() -> None:
    ap = base_argparser("Stage 4: plane cut, floater removal, decimation")
    ap.add_argument("--target-tris", type=int, default=120000,
                    help="web budget; brief asks for roughly 100k-150k")
    ap.add_argument("--plane-inlier-frac", type=float, default=0.18,
                    help="only cut the plane if it holds at least this share of vertices")
    ap.add_argument("--plane-thresh-ratio", type=float, default=0.004,
                    help="RANSAC distance threshold as a fraction of the bounding-box diagonal")
    ap.add_argument("--component-frac", type=float, default=0.05)
    ap.add_argument("--no-plane-cut", action="store_true")
    ap.add_argument("--force-plane-cut", action="store_true",
                    help="cut the plane even when densification was masked")
    args = ap.parse_args()

    p = Paths(args.dataset).ensure()
    if stage_done(p, "s4_cleanup") and not args.force:
        log("[s4_cleanup] already done (use --force to redo)")
        return

    src = p.mvs / "scene_mesh.ply"
    if not src.exists():
        raise RuntimeError(f"{src} missing — run s3_mvs first")
    dst = p.mvs / "scene_clean.ply"

    with Stage(p, "s4_cleanup") as st:
        import open3d as o3d
        m = load(src)
        v0, t0 = counts(m)
        st.notes["verts_before"], st.notes["tris_before"] = v0, t0
        log(f"  in: {v0} verts / {t0} tris")

        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()

        caveats = []
        # If densification ran masked, the table never entered the depth maps and
        # there is no plane to cut. Running RANSAC anyway is actively harmful on a
        # flat subject: on the test dish the dominant plane WAS the cookie's own
        # face, and cutting it removed 22.5% of the object. So the plane cut is a
        # backstop for the UNMASKED path only.
        masked_densify = bool(_note_from_state(p, "s3_mvs", "masks_used", False))
        if masked_densify and not args.force_plane_cut:
            args.no_plane_cut = True
            st.notes["plane_cut"] = (
                "skipped by design — s3_mvs densified with SAM 2 masks, so the "
                "table was never reconstructed and there is no plane to remove. "
                "Pass --force-plane-cut to override.")
            log("  plane cut skipped: densification was masked")

        if not args.no_plane_cut:
            m, cut = cut_plane(m, args.plane_thresh_ratio, args.plane_inlier_frac,
                               st.notes)
            if not cut:
                caveats.append("Table plane was NOT cut in s4_cleanup: "
                               + str(st.notes.get("plane_cut")))
        elif "plane_cut" not in st.notes:
            st.notes["plane_cut"] = "disabled by --no-plane-cut"

        m = keep_components(m, args.component_frac, st.notes)
        m = fill_holes(m, st.notes)

        v1, t1 = counts(m)
        if t1 > args.target_tris:
            m = m.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
            m.remove_degenerate_triangles()
            m.remove_unreferenced_vertices()
            st.notes["decimated"] = f"{t1} -> {len(m.triangles)}"
        else:
            st.notes["decimated"] = (f"skipped — {t1} tris already under the "
                                     f"{args.target_tris} target")

        m.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(dst), m)

        v2, t2 = counts(m)
        st.notes["verts_after"], st.notes["tris_after"] = v2, t2
        st.notes["output"] = str(dst)
        if caveats:
            st.notes["caveats"] = caveats
        log(f"  out: {v2} verts / {t2} tris -> {dst.name}")


if __name__ == "__main__":
    main()
