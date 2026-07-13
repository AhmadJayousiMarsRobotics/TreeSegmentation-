import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

import laspy
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter, maximum_filter, grey_dilation
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist
from skimage.segmentation import watershed

from .models import TreeRecord, ExtractionParams

logger = logging.getLogger("tree_extractor")

# ---------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------

def compute_cloud_stats_streaming(path: str, params: ExtractionParams) -> tuple[float, float, float, float]:
    """Stream first chunk to estimate density, spacing, etc."""
    with laspy.open(path) as f:
        header = f.header
        dx = float(header.maxs[0] - header.mins[0])
        dy = float(header.maxs[1] - header.mins[1])
        area = dx * dy
        n_all = header.point_count
        spacing_full = np.sqrt(area / n_all) if n_all > 1 else 0.1

        target_classes = set(params.classes)

        for chunk in f.chunk_iterator(2_000_000):
            cls = np.array(chunk.classification)

            if params.use_all:
                mask = cls != 2
            else:
                available = set(np.unique(cls))
                use_c = sorted(target_classes & available)
                if use_c:
                    mask = np.isin(cls, use_c)
                else:
                    mask = cls != 2
            n_tree_chunk = int(mask.sum())
            frac = n_tree_chunk / len(chunk) if len(chunk) > 0 else 0
            n_tree_est = int(frac * n_all)
            density_tree = n_tree_est / area if area > 0 else 1.0
            return area, spacing_full, density_tree, float(n_tree_est)
    return 0.0, 0.1, 1.0, 0.0

# ---------------------------------------------------------------------------
# Out-of-core tiling  (with shared DTM raster)
# ---------------------------------------------------------------------------

def _build_global_dtm_raster(
    ground_x: np.ndarray, ground_y: np.ndarray, ground_z: np.ndarray,
    x_min: float, y_min: float, x_max: float, y_max: float,
    resolution: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """
    Build a coarse global DTM raster from all ground points collected
    during streaming.  Returns the raster array and its metadata dict.
    """
    cols = max(1, int(np.ceil((x_max - x_min) / resolution)) + 1)
    rows = max(1, int(np.ceil((y_max - y_min) / resolution)) + 1)

    col_idx = np.clip(((ground_x - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((ground_y - y_min) / resolution).astype(int), 0, rows - 1)

    # Accumulate sum and count per cell for averaging
    dtm_sum = np.zeros((rows, cols), dtype=np.float64)
    dtm_cnt = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(dtm_sum, (row_idx, col_idx), ground_z)
    np.add.at(dtm_cnt, (row_idx, col_idx), 1.0)

    valid = dtm_cnt > 0
    dtm = np.full((rows, cols), np.nan, dtype=np.float64)
    dtm[valid] = dtm_sum[valid] / dtm_cnt[valid]

    # Fill empty cells using nearest neighbour interpolation
    if np.any(~valid):
        from scipy.interpolate import NearestNDInterpolator
        
        # Get coordinates of valid pixels
        valid_r, valid_c = np.where(valid)
        valid_z = dtm[valid_r, valid_c]
        
        if len(valid_z) > 0:
            interp = NearestNDInterpolator(np.column_stack([valid_r, valid_c]), valid_z)
            
            # Get coordinates of NaN pixels
            nan_r, nan_c = np.where(~valid)
            
            # Interpolate in chunks to avoid memory spikes
            chunk_size = 1_000_000
            for i in range(0, len(nan_r), chunk_size):
                end = min(i + chunk_size, len(nan_r))
                dtm[nan_r[i:end], nan_c[i:end]] = interp(nan_r[i:end], nan_c[i:end])

    meta = {
        "x_min": x_min, "y_min": y_min,
        "resolution": resolution,
        "rows": rows, "cols": cols,
    }
    return dtm, meta


def split_cloud_to_disk(
    path: str, temp_dir: Path, params: ExtractionParams,
) -> tuple[List[Path], Optional[tuple[np.ndarray, dict]], tuple[float, float]]:
    """
    Stream point cloud to spatially-tiled binary files on disk.

    Also collects ground points (class 2) to build a shared DTM raster,
    so individual tiles no longer need to include ground points or build
    their own DTM from scratch.

    Returns
    -------
    tiles : list[Path]
        Paths to the binary tile files.
    dtm_info : tuple[ndarray, dict] | None
        (dtm_raster, meta) if enough ground points were found, else None.
    tile_meta : tuple[float, float]
        (min_x, min_y) bounding box corner for core grid alignment.
    """
    logger.info("Streaming %s to disk tiles...", path)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing temp files if any
    for old_file in temp_dir.glob("tile_*.bin"):
        old_file.unlink()

    # Accumulators for ground points (for shared DTM)
    ground_xs: list[np.ndarray] = []
    ground_ys: list[np.ndarray] = []
    ground_zs: list[np.ndarray] = []

    with laspy.open(path) as f:
        header = f.header
        min_x, min_y = header.mins[0], header.mins[1]
        max_x, max_y = header.maxs[0], header.maxs[1]

        target_classes = set(params.classes)

        size = params.chunk_size
        buf = params.chunk_buffer

        total_pts = header.point_count
        processed = 0

        for chunk in f.chunk_iterator(5_000_000):
            processed += len(chunk)
            logger.info("  Streaming points: %s / %s...", f"{processed:,}", f"{total_pts:,}")
            cls = np.array(chunk.classification)


            # --- Collect ground points for shared DTM ---
            gnd_mask = cls == 2
            if np.any(gnd_mask):
                gx = np.array(chunk.x)[gnd_mask]
                gy = np.array(chunk.y)[gnd_mask]
                gz = np.array(chunk.z)[gnd_mask]
                
                # Prevent RAM exhaustion on massive files: cap ground points per chunk
                if len(gx) > 50_000:
                    rng = np.random.default_rng()
                    idx = rng.choice(len(gx), 50_000, replace=False)
                    gx, gy, gz = gx[idx], gy[idx], gz[idx]
                    
                ground_xs.append(gx)
                ground_ys.append(gy)
                ground_zs.append(gz)

            # --- Filter for vegetation points (no ground needed in tiles) ---
            if params.use_all:
                veg_mask = cls != 2
            else:
                available = set(np.unique(cls))
                use_c = target_classes & available
                if use_c:
                    veg_mask = np.isin(cls, list(use_c))
                else:
                    # Fallback to all non-ground points if requested classes are missing
                    veg_mask = cls != 2

            if not np.any(veg_mask):
                continue

            x = np.array(chunk.x)[veg_mask]
            y = np.array(chunk.y)[veg_mask]
            z = np.array(chunk.z)[veg_mask]
            c = cls[veg_mask]

            data = np.column_stack((x, y, z, c)).astype(np.float64)

            # Grid bounds for each point
            c_min_x = np.floor((x - buf - min_x) / size).astype(int)
            c_max_x = np.floor((x + buf - min_x) / size).astype(int)
            c_min_y = np.floor((y - buf - min_y) / size).astype(int)
            c_max_y = np.floor((y + buf - min_y) / size).astype(int)

            unique_cx = np.arange(c_min_x.min(), c_max_x.max() + 1)
            unique_cy = np.arange(c_min_y.min(), c_max_y.max() + 1)

            for cx in unique_cx:
                for cy in unique_cy:
                    cell_min_x = min_x + cx * size - buf
                    cell_max_x = min_x + (cx + 1) * size + buf
                    cell_min_y = min_y + cy * size - buf
                    cell_max_y = min_y + (cy + 1) * size + buf

                    cell_mask = (
                        (x >= cell_min_x) & (x <= cell_max_x) &
                        (y >= cell_min_y) & (y <= cell_max_y)
                    )
                    if np.any(cell_mask):
                        tile_path = temp_dir / f"tile_{cx}_{cy}.bin"
                        with open(tile_path, "ab") as tf:
                            data[cell_mask].tofile(tf)

    tiles = list(temp_dir.glob("tile_*.bin"))
    logger.info("Created %d spatial tiles on disk.", len(tiles))

    # --- Build shared DTM raster ---
    dtm_info = None
    if ground_xs:
        all_gx = np.concatenate(ground_xs)
        all_gy = np.concatenate(ground_ys)
        all_gz = np.concatenate(ground_zs)

        # Sub-sample if massive
        if len(all_gx) > 2_000_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_gx), 2_000_000, replace=False)
            all_gx, all_gy, all_gz = all_gx[idx], all_gy[idx], all_gz[idx]

        dtm_raster, dtm_meta = _build_global_dtm_raster(
            all_gx, all_gy, all_gz,
            float(min_x), float(min_y), float(max_x), float(max_y),
            resolution=2.0,
        )
        dtm_path = temp_dir / "dtm_raster.npy"
        np.save(dtm_path, dtm_raster)

        # Save meta alongside
        meta_path = temp_dir / "dtm_meta.npy"
        np.save(meta_path, np.array([
            dtm_meta["x_min"], dtm_meta["y_min"],
            dtm_meta["resolution"],
            float(dtm_meta["rows"]), float(dtm_meta["cols"]),
        ]))

        dtm_info = (dtm_raster, dtm_meta)
        logger.info(
            "Built shared DTM raster (%dx%d @ %.1fm) from %s ground points.",
            dtm_meta["rows"], dtm_meta["cols"],
            dtm_meta["resolution"], f"{len(all_gx):,}",
        )

    return tiles, dtm_info, (float(min_x), float(min_y))


# ---------------------------------------------------------------------------
# DTM / HAG  (shared raster or Cloth Simulation Filter fallback)
# ---------------------------------------------------------------------------

def _interpolate_dtm_raster(
    x: np.ndarray, y: np.ndarray,
    dtm_raster: np.ndarray, meta: dict,
) -> np.ndarray:
    """Bilinear interpolation of the shared DTM raster at arbitrary (x, y)."""
    res = meta["resolution"]
    x_min, y_min = meta["x_min"], meta["y_min"]
    rows, cols = meta["rows"], meta["cols"]

    fx = (x - x_min) / res
    fy = (y - y_min) / res

    c0 = np.clip(np.floor(fx).astype(int), 0, cols - 2)
    r0 = np.clip(np.floor(fy).astype(int), 0, rows - 2)

    sx = fx - c0
    sy = fy - r0

    ground_z = (
        dtm_raster[r0,     c0]     * (1 - sx) * (1 - sy) +
        dtm_raster[r0,     c0 + 1] *      sx  * (1 - sy) +
        dtm_raster[r0 + 1, c0 + 1] *      sx  *      sy  +
        dtm_raster[r0 + 1, c0]     * (1 - sx) *      sy
    )
    return ground_z


def build_dtm(x: np.ndarray, y: np.ndarray, z: np.ndarray,
              classification: np.ndarray) -> Callable:
    """
    Build a DTM surface interpolator from ground points.

    If sufficient class-2 ground points exist (>= 100), uses them directly.
    Otherwise runs a Cloth Simulation Filter (CSF) to classify ground
    from scratch.

    Returns a callable  dtm_interp(xi, yi) -> zi  that gives the exact
    interpolated ground elevation at arbitrary (x, y) coordinates.
    """
    mask = classification == 2

    if mask.sum() >= 100:
        gx, gy, gz = x[mask], y[mask], z[mask]
    else:
        # ---------------------------------------------------------------
        # Pure-Python Cloth Simulation Filter (CSF)
        # Faithful reimplementation of jianboqi/csf (Zhang et al. 2016).
        #
        # The algorithm inverts the point cloud Z-axis so the cloth
        # "falls" onto the terrain from above.  It uses Verlet
        # integration, precomputed constraint tables, permanent
        # particle pinning, and bilinear interpolation for the
        # final ground classification.
        #
        # VECTORIZED: the per-point rasterisation loop and empty-cell
        # fill have been replaced with fully vectorized NumPy operations.
        # ---------------------------------------------------------------

        # -- Parameters (matching C++ defaults) --
        cloth_res       = 1.0      # cloth grid resolution (m)
        rigidness       = 3        # constraint strength (index into LUT)
        time_step       = 0.65
        gravity         = 0.2
        class_thresh    = 0.5      # ground / off-ground distance
        n_iterations    = 500
        damping         = 0.01
        smooth_thresh   = 0.3      # slope-smooth height-diff threshold
        cloth_y_height  = 0.05
        clothbuffer     = 2        # grid cells of padding on each side

        # Precomputed constraint lookup tables (from Particle.h)
        single_move = [0, 0.3, 0.51, 0.657, 0.7599, 0.83193,
                       0.88235, 0.91765, 0.94235, 0.95965,
                       0.97175, 0.98023, 0.98616, 0.99031, 0.99322]
        double_move = [0, 0.3, 0.42, 0.468, 0.4872, 0.4949,
                       0.498, 0.4992, 0.4997, 0.4999,
                       0.4999, 0.5, 0.5, 0.5, 0.5]
        s1 = single_move[rigidness] if rigidness < 15 else 1.0
        d1 = double_move[rigidness] if rigidness < 15 else 0.5

        dt2 = time_step * time_step  # 0.4225

        # -- Coordinate transform (internal: y_int = -z_real) --
        neg_z = -z.astype(np.float64)

        x_min_c, x_max_c = float(x.min()), float(x.max())
        y_min_c, y_max_c = float(y.min()), float(y.max())
        neg_z_min, neg_z_max = float(neg_z.min()), float(neg_z.max())

        # -- Grid dimensions (with buffer) --
        n_cols = int(np.floor((x_max_c - x_min_c) / cloth_res)) + 2 * clothbuffer
        n_rows = int(np.floor((y_max_c - y_min_c) / cloth_res)) + 2 * clothbuffer
        n_cols = max(n_cols, 1)
        n_rows = max(n_rows, 1)

        origin_x = x_min_c - clothbuffer * cloth_res
        origin_y = y_min_c - clothbuffer * cloth_res
        cloth_init_h = neg_z_max + cloth_y_height

        # -- Vectorized rasterisation: nearest-point height per cell --
        col_i = np.round((x - origin_x) / cloth_res).astype(int)
        row_i = np.round((y - origin_y) / cloth_res).astype(int)
        col_i = np.clip(col_i, 0, n_cols - 1)
        row_i = np.clip(row_i, 0, n_rows - 1)

        cell_cx = origin_x + col_i * cloth_res
        cell_cy = origin_y + row_i * cloth_res
        sq_dist = (x - cell_cx) ** 2 + (y - cell_cy) ** 2

        # Sort by (cell, distance) so the closest point per cell comes first
        cell_linear = row_i * n_cols + col_i
        sort_idx = np.lexsort((sq_dist, cell_linear))
        sorted_cells = cell_linear[sort_idx]
        sorted_negz = neg_z[sort_idx]

        # First occurrence of each cell in the sorted array = closest point
        first_mask = np.empty(len(sorted_cells), dtype=bool)
        first_mask[0] = True
        first_mask[1:] = sorted_cells[1:] != sorted_cells[:-1]

        unique_cells = sorted_cells[first_mask]
        unique_negz = sorted_negz[first_mask]

        heightvals = np.full((n_rows, n_cols), -9999999999.0, dtype=np.float64)
        heightvals.ravel()[unique_cells] = unique_negz

        # -- Vectorized empty-cell fill via morphological dilation --
        EMPTY = -9999999999.0
        empty_mask = heightvals <= EMPTY
        if np.any(empty_mask):
            filled = heightvals.copy()
            for _ in range(max(n_rows, n_cols)):
                dilated = grey_dilation(filled, size=3)
                still_empty = filled <= EMPTY
                if not np.any(still_empty):
                    break
                filled[still_empty] = dilated[still_empty]
                # Stop if dilation didn't help (all neighbours empty too)
                if np.all(filled[still_empty] <= EMPTY):
                    break
            heightvals = filled

        # -- Cloth simulation (Verlet integration) --
        pos  = np.full((n_rows, n_cols), cloth_init_h, dtype=np.float64)
        old  = pos.copy()
        movable = np.ones((n_rows, n_cols), dtype=bool)

        accel_y = -gravity * dt2

        for _it in range(n_iterations):
            # 1. Verlet step on movable particles
            temp = pos.copy()
            displacement = (pos - old) * (1.0 - damping) + accel_y * dt2
            new_pos = np.where(movable, pos + displacement, pos)
            old_new = np.where(movable, temp, old)

            # 2. Constraint satisfaction (Y-component only)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if dr == -1:
                    n_pos = np.pad(new_pos[:-1, :], ((1, 0), (0, 0)), mode='edge')
                    n_mov = np.pad(movable[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=True)
                elif dr == 1:
                    n_pos = np.pad(new_pos[1:, :], ((0, 1), (0, 0)), mode='edge')
                    n_mov = np.pad(movable[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=True)
                elif dc == -1:
                    n_pos = np.pad(new_pos[:, :-1], ((0, 0), (1, 0)), mode='edge')
                    n_mov = np.pad(movable[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=True)
                else:
                    n_pos = np.pad(new_pos[:, 1:], ((0, 0), (0, 1)), mode='edge')
                    n_mov = np.pad(movable[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=True)

                diff = n_pos - new_pos

                both = movable & n_mov
                new_pos = np.where(both, new_pos + diff * d1, new_pos)

                self_only = movable & ~n_mov
                new_pos = np.where(self_only, new_pos + diff * s1, new_pos)

            pos = new_pos
            old = old_new

            # 3. Terrain collision — pin particles that touch
            touching = movable & (pos < heightvals)
            pos = np.where(touching, heightvals, pos)
            movable = movable & ~touching

            # 4. Convergence check
            max_diff = np.max(np.abs(pos[movable] - old[movable])) if np.any(movable) else 0.0
            if 0 < max_diff < 0.005:
                break

        # -- Slope smoothing (bSloopSmooth) --
        from scipy.ndimage import label as ndlabel
        movable_labels, n_components = ndlabel(movable.astype(np.int32))
        for comp_id in range(1, n_components + 1):
            comp_mask = movable_labels == comp_id
            if comp_mask.sum() <= 50:
                continue
            seeds = []
            rs, cs = np.where(comp_mask)
            for r, c in zip(rs, cs):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc_ = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc_ < n_cols:
                        if not movable[nr, nc_]:
                            if abs(heightvals[r, c] - heightvals[nr, nc_]) < smooth_thresh:
                                pos[r, c] = heightvals[r, c]
                                movable[r, c] = False
                                seeds.append((r, c))
                                break
            queue = list(seeds)
            head = 0
            while head < len(queue):
                cr, cc = queue[head]; head += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc_ = cr + dr, cc + dc
                    if 0 <= nr < n_rows and 0 <= nc_ < n_cols and movable[nr, nc_] and comp_mask[nr, nc_]:
                        if abs(heightvals[cr, cc] - heightvals[nr, nc_]) < smooth_thresh:
                            pos[nr, nc_] = heightvals[nr, nc_]
                            movable[nr, nc_] = False
                            queue.append((nr, nc_))

        # -- Cloud-to-cloth classification (bilinear interpolation) --
        fx = (x - origin_x) / cloth_res
        fy = (y - origin_y) / cloth_res
        c0 = np.clip(np.floor(fx).astype(int), 0, n_cols - 2)
        r0 = np.clip(np.floor(fy).astype(int), 0, n_rows - 2)
        sub_x = fx - c0
        sub_y = fy - r0

        cloth_h = (pos[r0,     c0]     * (1 - sub_x) * (1 - sub_y) +
                   pos[r0,     c0 + 1] *      sub_x  * (1 - sub_y) +
                   pos[r0 + 1, c0 + 1] *      sub_x  *      sub_y  +
                   pos[r0 + 1, c0]     * (1 - sub_x) *      sub_y)

        height_var = cloth_h - neg_z
        ground_mask = np.abs(height_var) < class_thresh

        if ground_mask.sum() < 10:
            z_min = float(np.min(z))
            return lambda xi, yi: np.full(len(xi), z_min, dtype=np.float64)

        gx, gy, gz = x[ground_mask], y[ground_mask], z[ground_mask]

    # Sub-sample ground points for interpolation performance
    if len(gx) > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(gx), 50_000, replace=False)
        gx, gy, gz = gx[idx], gy[idx], gz[idx]

    ground_xy = np.column_stack([gx, gy])
    linear_interp = LinearNDInterpolator(ground_xy, gz)
    nearest_interp = NearestNDInterpolator(ground_xy, gz)

    def dtm_interpolator(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        """Interpolate ground elevation; nearest-neighbour fills any NaN gaps."""
        query = np.column_stack([xi, yi])
        result = linear_interp(query)
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            result[nan_mask] = nearest_interp(query[nan_mask])
        return result

    return dtm_interpolator


def compute_hag(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                dtm_interp: Callable) -> np.ndarray:
    """
    Compute Height Above Ground by subtracting the exact interpolated
    DTM elevation directly beneath each specific (x, y) coordinate.
    """
    ground_z = dtm_interp(x, y)
    return z - ground_z

# ---------------------------------------------------------------------------
# Point filtering
# ---------------------------------------------------------------------------

def filter_points(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  classification: np.ndarray, hag: np.ndarray,
                  classes: tuple[int, ...], use_all: bool,
                  min_h: float) -> np.ndarray:
    """Filter to retain only vegetation points above minimum height."""
    if use_all:
        mask = (classification != 2) & (hag >= min_h) & np.isfinite(hag)
    else:
        available = set(np.unique(classification))
        use_c = sorted(set(classes) & available)
        if not use_c:
            mask = (classification != 2) & (hag >= min_h) & np.isfinite(hag)
        else:
            mask = np.isin(classification, use_c) & (hag >= min_h) & np.isfinite(hag)

    pts = np.column_stack([x[mask], y[mask], z[mask], hag[mask]])
    return pts

# ---------------------------------------------------------------------------
# Tree segmentation  (CHM raster + Variable-Window Local Max + Watershed)
# ---------------------------------------------------------------------------

def _variable_window_local_max(
    chm: np.ndarray, chm_resolution: float, min_height: float,
) -> np.ndarray:
    """
    Detect treetops using variable-window local maxima (Popescu & Wynne).

    The search window scales with the local CHM height using an allometric
    model: crown_radius = 0.1 * height + 1.0 (metres).  This avoids
    over-segmenting large trees (fixed small window) and under-segmenting
    small trees (fixed large window).

    Implemented efficiently by processing height bands and applying
    ``scipy.ndimage.maximum_filter`` with the appropriate window size
    per band.
    """
    rows, cols = chm.shape
    is_max = np.zeros((rows, cols), dtype=bool)

    h_max = float(chm.max())
    if h_max < min_height:
        return np.empty((0, 2), dtype=int)

    # Process in 1-metre height bands
    band_edges = np.arange(min_height, h_max + 1.5, 1.0)

    for i in range(len(band_edges) - 1):
        h_lo = band_edges[i]
        h_hi = band_edges[i + 1]
        h_mid = (h_lo + h_hi) / 2.0

        # Allometric crown radius → window size in pixels
        crown_r_m = 0.1 * h_mid + 1.0
        win_px = max(3, int(2 * crown_r_m / chm_resolution) | 1)  # ensure odd

        local_max = maximum_filter(chm, size=win_px)

        band_mask = (chm >= h_lo) & (chm < h_hi)
        is_max |= band_mask & (chm >= local_max)

    # Handle the highest band (>= last edge)
    if h_max >= band_edges[-1]:
        h_mid = (band_edges[-1] + h_max) / 2.0
        crown_r_m = 0.1 * h_mid + 1.0
        win_px = max(3, int(2 * crown_r_m / chm_resolution) | 1)
        local_max = maximum_filter(chm, size=win_px)
        band_mask = chm >= band_edges[-1]
        is_max |= band_mask & (chm >= local_max)

    coords = np.array(np.where(is_max)).T  # shape (N, 2)
    return coords


def segment_trees(pts: np.ndarray, chm_resolution: float,
                  min_height: float, min_points: int) -> List[np.ndarray]:
    """
    Segment individual trees using a CHM rasterisation → Variable-Window
    Local Maxima → marker-controlled Watershed pipeline.

    Steps
    -----
    1. Rasterise the normalised points into a 2D Canopy Height Model (CHM)
       at *chm_resolution* metres, storing the **maximum HAG** per pixel.
    2. Smooth the CHM with an adaptive Gaussian filter (σ scales with
       resolution to avoid over-/under-smoothing).
    3. Detect treetop candidates via variable-window local maxima where
       the search radius scales allometrically with local CHM height.
    4. Run a marker-controlled watershed on the **inverted** smooth CHM to
       delineate individual crown regions.
    5. Map each 3-D point back to its watershed label and group the results
       into distinct per-tree clusters.

    Parameters
    ----------
    pts : ndarray, shape (N, 4)
        Columns are (x, y, z, hag).
    chm_resolution : float
        Pixel size in metres for the CHM raster.
    min_height : float
        Minimum HAG threshold for peak detection.
    min_points : int
        Minimum points per tree segment.

    Returns
    -------
    list[ndarray]
        One (N_i, 4) array per segmented tree.
    """
    x, y, hag = pts[:, 0], pts[:, 1], pts[:, 3]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    cols = max(1, int(np.ceil((x_max - x_min) / chm_resolution)) + 1)
    rows = max(1, int(np.ceil((y_max - y_min) / chm_resolution)) + 1)

    # Map each point to its raster grid cell
    col_idx = np.clip(((x - x_min) / chm_resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((y - y_min) / chm_resolution).astype(int), 0, rows - 1)

    # Build CHM: maximum HAG value per pixel
    chm = np.zeros((rows, cols), dtype=np.float64)
    np.maximum.at(chm, (row_idx, col_idx), hag)

    # Adaptive sigma: scales with resolution to keep smoothing consistent
    sigma = max(0.5, min(2.0, 1.5 / chm_resolution))
    chm_smooth = gaussian_filter(chm, sigma=sigma)

    # Variable-window local maxima (allometric crown model)
    coords = _variable_window_local_max(chm_smooth, chm_resolution, min_height)

    if len(coords) == 0:
        if len(pts) >= min_points:
            return [pts]
        return []

    # Create marker array for watershed
    markers = np.zeros((rows, cols), dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1  # labels are 1-indexed

    # Marker-controlled watershed on inverted CHM
    canopy_mask = chm > min_height
    labels = watershed(-chm_smooth, markers, mask=canopy_mask)

    # Assign each point to its watershed label
    point_labels = labels[row_idx, col_idx]

    # Group points by label
    clusters: List[np.ndarray] = []
    for lbl in np.unique(point_labels):
        if lbl <= 0:  # skip background
            continue
        m = point_labels == lbl
        if m.sum() >= min_points:
            clusters.append(pts[m])

    return clusters

# ---------------------------------------------------------------------------
# Per-tree metric extraction  (3-D ConvexHull volume + Feret + DBH)
# ---------------------------------------------------------------------------

def extract_tree(cluster: np.ndarray, tree_id: int) -> TreeRecord:
    """
    Extract geometric metrics for a single segmented tree cluster.

    Improvements over baseline:
    * **Trunk position**: height-weighted centroid of the top-30% canopy
      points for robustness against understory noise.
    * **Crown diameter**: Feret diameter (max pairwise distance between
      2-D ConvexHull vertices) instead of axis-aligned bounding box.
    * **DBH**: allometric estimate from height and crown diameter
      (Jucker et al. 2017).
    * Crown **area** via 2-D ConvexHull.
    * Crown **volume** via 3-D ConvexHull with bounding-box fallback.
    """
    rec = TreeRecord(tree_id=tree_id, point_count=len(cluster))
    c = cluster
    x, y, z, hag = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

    rec.height_m = float(np.max(hag))
    if rec.height_m < ExtractionParams.min_height:
        return rec

    rec.z_ground = float(np.median(z - hag))

    # --- Trunk position: height-weighted centroid of top-30% points ---
    hag_thresh_70 = float(np.percentile(hag, 70))
    top_mask = hag >= hag_thresh_70
    if top_mask.sum() > 3:
        weights = hag[top_mask]
        rec.x = float(np.average(x[top_mask], weights=weights))
        rec.y = float(np.average(y[top_mask], weights=weights))
    else:
        rec.x, rec.y = float(np.mean(x)), float(np.mean(y))

    hag_crown = float(np.percentile(hag, 50))
    cm = hag >= hag_crown
    nc = int(cm.sum())

    if nc >= 3:
        cx, cy = x[cm], y[cm]
        crown_hag = hag[cm]

        # --- 2-D ConvexHull for crown area + Feret diameter ---
        if nc >= 5:
            try:
                pts_2d = np.column_stack([cx, cy])
                if nc > 500:
                    idx = np.linspace(0, nc - 1, 500, dtype=int)
                    hull = ConvexHull(pts_2d[idx])
                else:
                    hull = ConvexHull(pts_2d)
                rec.crown_area_m2 = float(hull.volume)  # 2-D hull .volume = area

                # Feret diameter: max distance between hull vertices
                hull_pts = pts_2d[hull.vertices]
                if len(hull_pts) >= 2:
                    dists = pdist(hull_pts)
                    rec.crown_diam_m = float(np.max(dists))
                else:
                    d = max(float(np.max(cx) - np.min(cx)),
                            float(np.max(cy) - np.min(cy)))
                    rec.crown_diam_m = d
            except Exception:
                d = max(float(np.max(cx) - np.min(cx)),
                        float(np.max(cy) - np.min(cy)))
                rec.crown_diam_m = d
                rec.crown_area_m2 = 0.25 * math.pi * d * d
        else:
            d = max(float(np.max(cx) - np.min(cx)),
                    float(np.max(cy) - np.min(cy)))
            rec.crown_diam_m = d
            rec.crown_area_m2 = 0.25 * math.pi * d * d

        # --- 3-D ConvexHull for crown volume ---
        if rec.crown_diam_m > 0 and rec.height_m > 0 and nc >= 4:
            crown_xyz = np.column_stack([cx, cy, crown_hag])
            try:
                if nc > 500:
                    idx_3d = np.linspace(0, nc - 1, 500, dtype=int)
                    hull_3d = ConvexHull(crown_xyz[idx_3d])
                else:
                    hull_3d = ConvexHull(crown_xyz)
                rec.crown_volume_m3 = float(hull_3d.volume)
            except Exception:
                bbox_dx = float(np.max(cx) - np.min(cx))
                bbox_dy = float(np.max(cy) - np.min(cy))
                bbox_dz = float(np.max(crown_hag) - np.min(crown_hag))
                rec.crown_volume_m3 = bbox_dx * bbox_dy * max(bbox_dz, 0.1)

    # --- DBH allometric estimate (Jucker et al. 2017) ---
    if rec.height_m > 0 and rec.crown_diam_m > 0:
        rec.dbh_cm = round(
            1.2 * (rec.height_m ** 0.74) * (rec.crown_diam_m ** 0.26), 2
        )

    return rec

# ---------------------------------------------------------------------------
# Tile processing
# ---------------------------------------------------------------------------

def process_chunk(
    tile_path: Path, params: ExtractionParams, tile_idx: int,
    dtm_raster_path: Optional[str] = None,
    dtm_meta_path: Optional[str] = None,
    tile_meta: Optional[tuple[float, float]] = None
) -> tuple[List[TreeRecord], Optional[List[np.ndarray]]]:
    """
    Process a single spatial tile.

    If a shared DTM raster is available (paths provided), interpolates HAG
    directly from it.  Otherwise falls back to building a per-tile DTM
    (CSF if no ground points are present).

    Returns
    -------
    records : list[TreeRecord]
    segments : list[ndarray] | None
        Per-tree point arrays if export_segments is True in params.
    """
    try:
        # Memory-mapped tile reading to reduce peak RAM
        raw = np.memmap(tile_path, dtype=np.float64, mode='r')
        if len(raw) == 0:
            return [], None
        data = np.array(raw.reshape(-1, 4))  # copy into regular array
        del raw  # release mmap

        x, y, z, cls = data[:, 0], data[:, 1], data[:, 2], data[:, 3].astype(np.uint8)

        # --- Compute HAG from shared DTM or per-tile fallback ---
        if dtm_raster_path and dtm_meta_path:
            dtm_raster = np.load(dtm_raster_path)
            meta_arr = np.load(dtm_meta_path)
            meta = {
                "x_min": float(meta_arr[0]),
                "y_min": float(meta_arr[1]),
                "resolution": float(meta_arr[2]),
                "rows": int(meta_arr[3]),
                "cols": int(meta_arr[4]),
            }
            ground_z = _interpolate_dtm_raster(x, y, dtm_raster, meta)
            hag = z - ground_z
        else:
            # Fallback: build per-tile DTM (requires ground points in tile)
            dtm_interp = build_dtm(x, y, z, cls)
            hag = compute_hag(x, y, z, dtm_interp)

        pts = filter_points(x, y, z, cls, hag, params.classes, params.use_all, params.min_height)

        if len(pts) == 0:
            return [], None

        if params.subsample and 0 < params.subsample < 1:
            n0 = len(pts)
            pts = pts[np.random.default_rng(0).random(n0) < params.subsample]

        clusters = segment_trees(pts, params.chm_resolution, params.min_height,
                                 params.min_points)

        core_min_x, core_max_x = -np.inf, np.inf
        core_min_y, core_max_y = -np.inf, np.inf
        
        if tile_meta is not None:
            min_x, min_y = tile_meta
            import re
            m = re.search(r"tile_(-?\d+)_(-?\d+)\.bin", Path(tile_path).name)
            if m:
                cx, cy = int(m.group(1)), int(m.group(2))
                core_min_x = min_x + cx * params.chunk_size
                core_max_x = min_x + (cx + 1) * params.chunk_size
                core_min_y = min_y + cy * params.chunk_size
                core_max_y = min_y + (cy + 1) * params.chunk_size

        records: List[TreeRecord] = []
        segments: List[np.ndarray] = [] if params.export_segments else None
        
        for i, cl in enumerate(clusters):
            # Base ID offset per tile to avoid duplicates before NMS
            cid = (tile_idx * 100_000) + (i + 1)
            r = extract_tree(cl, cid)
            if params.min_height <= r.height_m <= params.max_height and r.crown_diam_m >= 1.0:
                # Discard trees whose centroid falls in the buffer zone of this tile
                if core_min_x <= r.x < core_max_x and core_min_y <= r.y < core_max_y:
                    records.append(r)
                    if segments is not None:
                        segments.append(cl)
                        
        return records, segments
    except Exception as e:
        logger.error("Error processing %s: %s", tile_path, e)
        return [], None

# ---------------------------------------------------------------------------
# Non-Maximum Suppression  (proper circle-circle IoU)
# ---------------------------------------------------------------------------

def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """Compute the intersection area of two circles with radii r1, r2
    whose centres are separated by distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    # Standard lens-area formula
    part1 = r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    part2 = r2 ** 2 * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    part3 = 0.5 * math.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    return part1 + part2 - part3


def apply_nms(records: List[TreeRecord], segments: Optional[List[np.ndarray]], nms_dist: float,
              iou_threshold: float = 0.3) -> tuple[List[TreeRecord], Optional[List[np.ndarray]]]:
    """
    Non-Maximum Suppression to remove duplicate trees from tile overlaps.

    Uses proper circle-circle IoU (crown radii as circles) instead of
    simplified distance heuristics.
    """
    if not records:
        return [], segments
        
    logger.info("Applying NMS to %d candidate trees...", len(records))
    
    if segments is not None:
        paired = list(zip(records, segments))
        paired.sort(key=lambda x: x[0].height_m, reverse=True)
        records = [p[0] for p in paired]
        segments = [p[1] for p in paired]
    else:
        records.sort(key=lambda r: r.height_m, reverse=True)

    coords = np.array([[r.x, r.y] for r in records])
    tree = KDTree(coords)

    kept_recs = []
    kept_segs = [] if segments is not None else None
    suppressed = set()

    for i, r in enumerate(records):
        if i in suppressed:
            continue
        kept_recs.append(r)
        if segments is not None:
            kept_segs.append(segments[i])

        idx = tree.query_ball_point([r.x, r.y], nms_dist)
        for j in idx:
            if j != i and j not in suppressed:
                n = records[j]
                dist = math.hypot(r.x - n.x, r.y - n.y)

                # Near-identical positions → always suppress
                if dist < 0.5:
                    suppressed.add(j)
                    continue

                r1 = r.crown_diam_m / 2.0
                r2 = n.crown_diam_m / 2.0

                if r1 <= 0 or r2 <= 0 or dist >= r1 + r2:
                    continue

                # Proper circle-circle IoU
                inter = _circle_intersection_area(r1, r2, dist)
                area1 = math.pi * r1 ** 2
                area2 = math.pi * r2 ** 2
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0.0

                if iou > iou_threshold:
                    suppressed.add(j)

    logger.info("NMS kept %d trees (suppressed %d)", len(kept_recs), len(records) - len(kept_recs))
    for i, r in enumerate(kept_recs):
        r.tree_id = i + 1
        
    return kept_recs, kept_segs

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_point_cloud(
    path: str, params: ExtractionParams,
    progress_callback: Optional[Callable] = None,
) -> tuple[List[TreeRecord], Optional[List[np.ndarray]]]:
    """
    Process a LAS/LAZ file end-to-end and return extracted tree records.

    Parameters
    ----------
    path : str
        Path to the input LAS/LAZ file.
    params : ExtractionParams
        Processing parameters.
    progress_callback : callable, optional
        Called with (done, total) after each tile completes.

    Returns
    -------
    records : list[TreeRecord]
    all_segments : list[ndarray] | None
        Per-tree point arrays if export_segments is enabled.
    """
    if params.auto:
        _, spacing, density, n_tree = compute_cloud_stats_streaming(path, params)
        params.chm_resolution = max(0.25, min(1.0, 2.0 * spacing))
        params.min_points = max(15, int(math.pi * (params.chm_resolution / 2) ** 2 * density / 2))
        logger.info(
            "Auto: spacing=%.3fm, est. density=%.0f pts/m2  -> chm_res=%.2f, min_pts=%d",
            spacing, density, params.chm_resolution, params.min_points,
        )

    stem = Path(path).stem
    temp_dir = Path(path).parent / f"{stem}_tiles_temp"

    tiles, dtm_info, tile_meta = split_cloud_to_disk(path, temp_dir, params)

    nc = len(tiles)
    logger.info("Extracting metrics from %d tiles...", nc)

    # Shared DTM paths for child processes
    dtm_raster_path = str(temp_dir / "dtm_raster.npy") if dtm_info else None
    dtm_meta_path = str(temp_dir / "dtm_meta.npy") if dtm_info else None

    all_records: List[TreeRecord] = []
    all_segments: Optional[List[np.ndarray]] = [] if params.export_segments else None

    nw = params.workers or min(os.cpu_count() or 4, nc, 16)
    logger.info("Workers: %d", nw)

    batch = [
        (t, params, i, dtm_raster_path, dtm_meta_path, tile_meta)
        for i, t in enumerate(tiles)
    ]

    if nw > 1:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(process_chunk, *b) for b in batch]
            done = 0
            for f in as_completed(futs):
                recs, segs = f.result()
                all_records.extend(recs)
                if all_segments is not None and segs:
                    all_segments.extend(segs)
                done += 1
                if progress_callback:
                    progress_callback(done, nc)
                elif done % max(1, nc // 10) == 0:
                    logger.info("  [%d/%d] tiles processed...", done, nc)
    else:
        for i, b in enumerate(batch):
            recs, segs = process_chunk(*b)
            all_records.extend(recs)
            if all_segments is not None and segs:
                all_segments.extend(segs)
            if progress_callback:
                progress_callback(i + 1, nc)
            elif (i + 1) % max(1, nc // 10) == 0:
                logger.info("  [%d/%d] tiles processed...", i + 1, nc)

    logger.info("Aggregated %d trees before NMS.", len(all_records))

    # Cleanup tiles
    for t in tiles:
        t.unlink(missing_ok=True)
    if dtm_raster_path:
        Path(dtm_raster_path).unlink(missing_ok=True)
    if dtm_meta_path:
        Path(dtm_meta_path).unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
        logger.info("Cleaned up temporary tiles.")
    except Exception:
        pass

    final_records, final_segments = apply_nms(all_records, all_segments, params.nms_dist)

    return final_records, final_segments
