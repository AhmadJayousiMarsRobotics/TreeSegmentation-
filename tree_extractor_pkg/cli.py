import argparse
import logging
import re
import sys
import time
from pathlib import Path

from .models import ExtractionParams
from .io import detect_epsg, write_csv, write_geojson, write_gpkg, write_segments, write_crown_polygons, write_report
from .core import process_point_cloud

logger = logging.getLogger("tree_extractor")


def _setup_logging(verbose: bool = False):
    """Configure the tree_extractor logger with a clean console handler."""
    log = logging.getLogger("tree_extractor")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("  %(message)s"))
        log.addHandler(handler)


def main():
    p = argparse.ArgumentParser(description="Extract individual trees from LAS/LAZ point clouds.")
    p.add_argument("input", help="Input .las / .laz file")
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--min-height", type=float, default=1.5,
                   help="Minimum HAG to consider (m)  (default: 1.5)")
    p.add_argument("--max-height", type=float, default=30.0,
                   help="Maximum tree height (m)  (default: 30.0)")
    p.add_argument("--auto", action="store_true",
                   help="Auto-detect eps & min_points from point density")
    p.add_argument("--chm-res", type=float, default=0.5,
                   help="CHM raster pixel size in metres  (default: 0.5)")
    p.add_argument("--min-points", type=int, default=15,
                   help="Minimum points per cluster  (default: 15)")
    p.add_argument("--all", action="store_true",
                   help="Use all non-ground points (ignore classification)")
    p.add_argument("--classes", type=str, default="3,4,5",
                   help="Tree classification classes (default: 3,4,5)")
    p.add_argument("--no-report", action="store_true", help="Skip text report")
    p.add_argument("--workers", type=int, default=0,
                   help="Extraction workers (0=auto, 1=sequential)")
    p.add_argument("--format", type=str, default="csv,gpkg",
                   help="Output formats: csv,gpkg,geojson (default: csv,gpkg)")
    p.add_argument("--epsg", type=int, default=None, help="Override EPSG for GPKG")
    p.add_argument("--subsample", type=float, default=0,
                   help="Random subsample fraction (e.g. 0.2) for dense clouds")

    p.add_argument("--chunk-size", type=float, default=200.0,
                   help="Out-of-core spatial tile size in meters (default: 200.0)")
    p.add_argument("--chunk-buffer", type=float, default=5.0,
                   help="Tile overlap buffer in meters (default: 5.0)")
    p.add_argument("--nms-dist", type=float, default=2.0,
                   help="Non-Maximum Suppression search radius in meters (default: 2.0)")

    p.add_argument("--export-segments", action="store_true",
                   help="Export a single merged segments.laz (color by tree_id) and crown_polygons.gpkg")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose (DEBUG-level) logging")
    args = p.parse_args()

    _setup_logging(verbose=args.verbose)

    classes = tuple(int(c.strip()) for c in args.classes.split(","))
    infile = Path(args.input)
    if not infile.exists():
        logger.error("Error: %s not found", infile)
        sys.exit(1)

    stem = re.sub(r"\.las|\.laz$", "", infile.stem)

    # Create an output directory named after the file
    base_outdir = Path(args.output_dir) if args.output_dir else infile.parent
    outdir = base_outdir / stem
    outdir.mkdir(parents=True, exist_ok=True)

    params = ExtractionParams(
        min_height=args.min_height,
        max_height=args.max_height,
        auto=args.auto,
        chm_resolution=args.chm_res,
        min_points=args.min_points,
        use_all=args.all,
        classes=classes,
        workers=args.workers,
        subsample=args.subsample,
        chunk_size=args.chunk_size,
        chunk_buffer=args.chunk_buffer,
        nms_dist=args.nms_dist,
        export_segments=args.export_segments,
    )

    t0 = time.time()

    # --- Progress bar with tqdm (optional) ---
    progress_callback = None
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=0, desc="Processing tiles", unit="tile", dynamic_ncols=True)

        def _progress(done: int, total: int):
            if pbar.total != total:
                pbar.total = total
                pbar.refresh()
            pbar.update(1)

        progress_callback = _progress
    except ImportError:
        pass  # tqdm not installed, fall back to log messages

    records, segments = process_point_cloud(str(infile), params,
                                            progress_callback=progress_callback)

    if pbar is not None:
        pbar.close()

    if not records:
        logger.info("No trees extracted.")
        sys.exit(0)

    epsg = args.epsg or detect_epsg(str(infile))
    if epsg:
        logger.info("CRS: EPSG:%d", epsg)

    fmts = [f.strip().lower() for f in args.format.split(",")]
    if "csv" in fmts:
        write_csv(records, str(outdir / "trees.csv"))
    if "geojson" in fmts:
        write_geojson(records, str(outdir / "trees.geojson"))
    if "gpkg" in fmts:
        write_gpkg(records, str(outdir / "trees.gpkg"), epsg)

    if params.export_segments and segments:
        write_segments(records, segments, str(outdir))
        write_crown_polygons(records, segments, str(outdir / "crowns.gpkg"), epsg)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f s", elapsed)

    if not args.no_report:
        report_str = write_report(records, str(infile), elapsed)
        print(report_str)
        # Also save the report to a text file for convenience
        with open(outdir / "report.txt", "w", encoding="utf-8") as f:
            f.write(report_str)

    logger.info("Done.")

if __name__ == "__main__":
    main()
