import csv
import json
import logging
import re
from dataclasses import fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import laspy

from .models import TreeRecord

logger = logging.getLogger("tree_extractor")


def detect_epsg(path: str) -> int:
    with laspy.open(path) as f:
        for vlr in f.header.vlrs:
            raw = str(vlr)
            m = re.search(r"UTM zone (\d+)([NS])?", raw)
            if m:
                z, h = int(m.group(1)), (m.group(2) or "N").upper()
                return 32600 + z if h == "N" else 32700 + z
    return 0


def write_csv(records: List[TreeRecord], path: str):
    fnames = [f.name for f in fields(TreeRecord)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fnames)
        for r in records:
            w.writerow([getattr(r, fn) for fn in fnames])
    logger.info("CSV -> %s  (%d trees)", path, len(records))


def write_geojson(records: List[TreeRecord], path: str):
    features = []
    for r in records:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.x, r.y, r.z_ground]},
            "properties": {
                "tree_id": r.tree_id,
                "height_m": round(r.height_m, 2),
                "crown_diam_m": round(r.crown_diam_m, 2),
                "crown_area_m2": round(r.crown_area_m2, 2),
                "crown_vol_m3": round(r.crown_volume_m3, 2),
                "dbh_cm": round(r.dbh_cm, 2),
                "point_cnt": r.point_count,
            },
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    logger.info("GeoJSON -> %s  (%d trees)", path, len(records))


def write_gpkg(records: List[TreeRecord], path: str, epsg: int = 0):
    import fiona  # type: ignore
    schema = {
        "geometry": "Point",
        "properties": {f.name: "float" for f in fields(TreeRecord)},
    }
    crs_str = f"EPSG:{epsg}" if epsg else None
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs=crs_str) as dst:
        for r in records:
            dst.write({
                "geometry": {"type": "Point", "coordinates": (r.x, r.y, r.z_ground)},
                "properties": {f.name: getattr(r, f.name) for f in fields(TreeRecord)},
            })
    logger.info("GeoPackage -> %s  (%d trees)", path, len(records))


def write_segments(
    records: List[TreeRecord],
    segments: List[np.ndarray],
    out_dir: str,
):
    """
    Export all segmented trees as a **single merged LAZ file** with
    ``tree_id`` and ``hag`` extra dimensions.

    Load one file into QGIS, then style/colorise by the ``tree_id``
    attribute to see every tree in a different colour.
    """
    # Merge all segments into one array, tagging each point with its tree_id
    all_x, all_y, all_z, all_hag, all_tid = [], [], [], [], []
    all_r, all_g, all_b = [], [], []
    
    for rec, seg in zip(records, segments):
        x, y, z, hag = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        all_hag.append(hag)
        all_tid.append(np.full(len(x), rec.tree_id, dtype=np.int32))
        
        # Generate a distinct random 16-bit RGB color for this tree
        rng = np.random.default_rng(rec.tree_id)
        all_r.append(np.full(len(x), rng.integers(10000, 65535), dtype=np.uint16))
        all_g.append(np.full(len(x), rng.integers(10000, 65535), dtype=np.uint16))
        all_b.append(np.full(len(x), rng.integers(10000, 65535), dtype=np.uint16))

    all_x  = np.concatenate(all_x)
    all_y  = np.concatenate(all_y)
    all_z  = np.concatenate(all_z)
    all_hag = np.concatenate(all_hag)
    all_tid = np.concatenate(all_tid)
    all_r  = np.concatenate(all_r)
    all_g  = np.concatenate(all_g)
    all_b  = np.concatenate(all_b)

    # Use point_format=3 which supports RGB colors natively
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = [float(np.min(all_x)), float(np.min(all_y)), float(np.min(all_z))]
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32,
                                                 description="Segmented tree ID"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="hag", type=np.float64,
                                                 description="Height Above Ground"))

    las = laspy.LasData(header)
    las.x = all_x
    las.y = all_y
    las.z = all_z
    las.tree_id = all_tid
    las.hag = all_hag
    las.red = all_r
    las.green = all_g
    las.blue = all_b

    fpath = Path(out_dir) / "segments.laz"
    las.write(str(fpath))
    logger.info("Segments LAZ -> %s  (%d trees, %s pts)",
                fpath, len(records), f"{len(all_x):,}")


def write_crown_polygons(
    records: List[TreeRecord],
    segments: List[np.ndarray],
    path: str,
    epsg: int = 0,
):
    """
    Export 2-D convex-hull crown polygons as a GeoPackage.

    Each polygon represents a tree's crown footprint computed from
    the upper-canopy points (above the 50th percentile HAG).  All
    tree metrics are stored as feature attributes.

    Drag-and-drop into QGIS for instant crown delineation overlay.
    """
    import fiona  # type: ignore
    from scipy.spatial import ConvexHull

    schema = {
        "geometry": "Polygon",
        "properties": {
            "tree_id": "int",
            "height_m": "float",
            "crown_diam_m": "float",
            "crown_area_m2": "float",
            "crown_vol_m3": "float",
            "dbh_cm": "float",
            "point_cnt": "int",
        },
    }
    crs_str = f"EPSG:{epsg}" if epsg else None

    with fiona.open(path, "w", driver="GPKG", schema=schema, crs=crs_str) as dst:
        for rec, seg in zip(records, segments):
            x, y, hag = seg[:, 0], seg[:, 1], seg[:, 3]

            # Upper-canopy points (above 50th percentile) for crown polygon
            hag_thresh = float(np.percentile(hag, 50))
            cm = hag >= hag_thresh
            cx, cy = x[cm], y[cm]

            if len(cx) < 3:
                # Too few points for a hull, trigger fallback
                ring = None
            else:
                try:
                    pts_2d = np.column_stack([cx, cy])
                    # Add tiny jitter to prevent collinearity crashes
                    pts_2d += np.random.normal(0, 0.005, pts_2d.shape)
                    
                    if len(pts_2d) > 500:
                        idx = np.linspace(0, len(pts_2d) - 1, 500, dtype=int)
                        hull = ConvexHull(pts_2d[idx])
                        ring = pts_2d[idx][hull.vertices]
                    else:
                        hull = ConvexHull(pts_2d)
                        ring = pts_2d[hull.vertices]
                except Exception:
                    ring = None

            # Fallback for sparse/collinear trees: draw an octagon around centroid
            if ring is None:
                r = rec.crown_diam_m / 2.0
                if r <= 0: r = 1.0  # Default 1m radius if diameter is 0
                cx_c, cy_c = rec.x, rec.y
                ring = [
                    (cx_c + r * np.cos(a), cy_c + r * np.sin(a))
                    for a in np.linspace(0, 2 * np.pi, 9)[:-1]
                ]

            # Close the ring (first vertex = last vertex)
            coords = [tuple(p) for p in ring]
            coords.append(coords[0])

            dst.write({
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "tree_id": rec.tree_id,
                    "height_m": round(rec.height_m, 2),
                    "crown_diam_m": round(rec.crown_diam_m, 2),
                    "crown_area_m2": round(rec.crown_area_m2, 2),
                    "crown_vol_m3": round(rec.crown_volume_m3, 2),
                    "dbh_cm": round(rec.dbh_cm, 2),
                    "point_cnt": rec.point_count,
                },
            })

    logger.info("Crown polygons -> %s  (%d trees)", path, len(records))


def write_report(records: List[TreeRecord], input_path: str, elapsed: float) -> str:
    v = [r for r in records if r.height_m > 0]
    h = [r.height_m for r in v]
    c = [r.crown_diam_m for r in v if r.crown_diam_m > 0]
    d = [r.dbh_cm for r in v if r.dbh_cm > 0]
    lines = [
        "=" * 72,
        "TREE EXTRACTION REPORT",
        "=" * 72,
        f"Input:   {input_path}",
        f"Time:    {elapsed:.1f} s",
        "",
        f"Trees found:           {len(v)}",
        "",
        "-- Height (m) --",
        f"  Min     {min(h):.2f}" if h else "",
        f"  Max     {max(h):.2f}" if h else "",
        f"  Mean    {np.mean(h):.2f}" if h else "",
        f"  Median  {np.median(h):.2f}" if h else "",
        "",
        "-- Crown diam (m) --",
        f"  Min     {min(c):.2f}" if c else "",
        f"  Max     {max(c):.2f}" if c else "",
        f"  Mean    {np.mean(c):.2f}" if c else "",
        "",
        "-- Est. DBH (cm) --",
        f"  Min     {min(d):.1f}" if d else "",
        f"  Max     {max(d):.1f}" if d else "",
        f"  Mean    {np.mean(d):.1f}" if d else "",
        "",
    ]
    # Top 10
    lines += [
        "-- Top 10 tallest --",
        f"{'ID':>5}  {'X':>14}  {'Y':>14}  {'Ht':>6}  {'Crown':>6}  {'DBH':>6}  {'Pts':>6}",
        "-" * 72,
    ]
    for r in sorted(v, key=lambda x: x.height_m, reverse=True)[:10]:
        lines.append(
            f"{r.tree_id:>5}  {r.x:>14.3f}  {r.y:>14.3f}  "
            f"{r.height_m:>6.2f}  {r.crown_diam_m:>6.2f}  {r.dbh_cm:>6.1f}  {r.point_count:>6}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)
