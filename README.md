# Tree Extractor

An optimized, out-of-core Python pipeline for extracting individual tree metrics from massive `.las` and `.laz` point clouds. This project provides a fully automated approach to single-tree segmentation and structural metric estimation (Height, Crown Diameter, Volume, DBH).

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage & CLI Arguments](#usage--cli-arguments)
4. [Project Architecture](#project-architecture)
5. [Detailed Function Logic](#detailed-function-logic)
6. [Algorithm Details](#algorithm-details)
7. [Output Structure & Formats](#output-structure--formats)
---

## Features

* **Out-of-Core Processing**: Can process massive point clouds without exhausting RAM by spatially chunking and streaming the file directly from disk.
* **Multi-processing**: Processes spatial tiles in **true parallel** across all available CPU cores using `ProcessPoolExecutor`.
* **Shared DTM Raster**: Builds a single global DTM during streaming — individual tiles interpolate Height Above Ground (HAG) from it instead of rebuilding ground models from scratch.
* **Variable-Window Local Maxima**: Detects treetops using allometric crown models (Popescu & Wynne) where the search window dynamically scales with the local canopy height.
* **Marker-Controlled Watershed**: Delineates individual crown regions natively from the Canopy Height Model (CHM).
* **Smart NMS Deduplication**: Uses Non-Maximum Suppression with proper circle-circle Intersection over Union (IoU) checks to merge overlapping chunks on the grid borders without deleting dense adjacent trees.
* **Feret Crown Diameter**: Measures crown diameter as the maximum pairwise distance between 2D convex hull vertices, avoiding the inaccuracy of axis-aligned bounding boxes.
* **DBH Estimation**: Allometric Diameter at Breast Height (DBH) estimation from height and crown diameter (Jucker et al. 2017).
* **Per-Tree Segment Export**: Optionally exports each tree's segmented point cloud as individual `.laz` files.
* **Multi-Format Export**: Extracts metrics into `CSV`, `GeoPackage`, and `GeoJSON` formats, including crown polygons.

---

## Installation

1. Create a virtual environment (recommended to isolate dependencies):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate     
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage & CLI Arguments

To run the extractor on a point cloud, simply use the `main.py` launcher:

```bash
python main.py path/to/cloud.laz --auto
```

This will automatically calculate the best clustering parameters (`eps` and `min_points`) based on the point density, segment the trees, and generate a dedicated output folder containing your extracted tree geometries and a detailed text report.

### Advanced Arguments

You can fully customize the extraction pipeline for advanced workloads using command-line arguments:

```bash
python main.py [input_file] [options]
```

**Core Options:**
* `--auto`: Automatically detect spacing and cluster minimum points from the overall point density.
* `--min-height`: Minimum Height Above Ground (HAG) to consider a tree (default: `1.5`m).
* `--max-height`: Maximum tree height threshold (default: `10.0`m).
* `--classes`: Comma-separated list of LAS classification values to extract as vegetation (default: `3,4,5`).
* `--all`: Ignore classifications and extract from all non-ground points.
* `--workers`: Number of parallel workers to use (`0` = auto-detect all available cores, default: `0`).

**Out-of-Core Tiling Options:**
* `--chunk-size`: Size of the spatial grid tiles in meters (default: `200.0`). Smaller values require less memory per core but increase border deduplication overhead.
* `--chunk-buffer`: Overlap buffer for the chunks in meters (default: `5.0`). Ensures trees on chunk borders are fully captured.
* `--nms-dist`: Non-Maximum Suppression (NMS) duplicate search radius for chunk boundaries in meters (default: `2.0`).

**Output Options:**
* `--output-dir`: Specific directory to save the output folder. If unset, a folder named after the input file is created in the same directory.
* `--format`: Comma-separated output formats to generate (default: `csv,gpkg`). Available: `csv`, `gpkg`, `geojson`.
* `--epsg`: Override the automatically detected EPSG coordinate system code.
* `--export-segments`: Export per-tree segmented point clouds `segments.laz`.
* `--no-report`: Do not print or save the text report `report.txt`.
* `--verbose` / `-v`: Enable verbose (DEBUG-level) logging.

---

## Project Architecture

The logic has been strictly decoupled into the `tree_extractor_pkg/` Python package:

* **`models.py`**: Contains simple dataclasses like `TreeRecord` for storing parsed metrics and `ExtractionParams` for encapsulating CLI arguments.
* **`io.py`**: Handles all disk operations: coordinate system detection, point cloud file streaming, multi-format spatial writes (CSV, GeoPackage, GeoJSON), and report generation.
* **`core.py`**: The mathematical and algorithmic engine. Handles out-of-core streaming, shared global DTM creation, coordinate interpolation, variable-window segmentation, parallelization mapping, and boundary deduplication (NMS).
* **`cli.py`**: Parses command-line inputs using `argparse`, configures logging, integrates `tqdm` progress bars, and bridges to `core.py`.

---

## Detailed Function Logic

### `core.py`

This module contains the heavy lifting for point cloud manipulation and geometric analysis:

* `compute_cloud_stats_streaming(path, params)`: Streams the `.laz` file dynamically without loading it fully into RAM to determine global bounds (Min/Max X, Y, Z), point counts, and overall density.
* `_build_global_dtm_raster(...)`: Constructs a singular global Digital Terrain Model (DTM) from pre-classified ground points (Class 2). This avoids computing ground models individually for every chunk.
* `split_cloud_to_disk(...)`: Discretizes the massive point cloud into smaller spatial grid chunks (e.g. 200m x 200m) with overlaps, caching them locally as temporary files for parallel processing.
* `build_dtm(...)`: A fallback routine utilizing the Cloth Simulation Filter (CSF) to create a ground model dynamically if the input cloud has no classified ground points.
* `_interpolate_dtm_raster(...)` & `compute_hag(...)`: Interpolates the ground elevation for arbitrary X/Y coordinates using the global DTM, allowing the conversion of absolute Z elevations into Height Above Ground (HAG).
* `filter_points(...)`: Strips away structural/ground points, retaining only valid vegetation classes within the specified `--min-height` and `--max-height` constraints.
* `_variable_window_local_max(...)`: A specialized local maxima algorithm. Instead of using a fixed-size search window to find treetops, the window size dynamically expands proportional to the canopy height, conforming to allometric crown equations.
* `segment_trees(...)`: Projects the 3D points into a 2D Canopy Height Model (CHM), applies scale-dependent Gaussian smoothing, runs the variable-window maxima, and segments individual crowns using a marker-controlled watershed algorithm.
* `extract_tree(...)`: Once a tree is delineated, this function calculates its geometric metrics. For example, it computes the 2D convex hull to find the maximum Feret Diameter and evaluates the Jucker et al. allometric equation for DBH.
* `process_chunk(...)`: The worker function executed in parallel. For a specific spatial tile, it orchestrates DTM extraction, HAG computation, segmentation, metric extraction, and returns the compiled `TreeRecord` instances.
* `apply_nms(...)`: Since spatial chunks overlap, trees on boundaries might be detected twice. This applies Non-Maximum Suppression, calculating Circle-Circle Intersection over Union (IoU) to discard duplicate detections while preserving naturally adjacent trees.
* `process_point_cloud(...)`: The main pipeline orchestrator. It manages the lifecycle of the entire extraction process: streaming stats -> chunking -> parallel processing -> NMS deduplication -> output generation.

### `io.py`

Handles robust reading/writing of spatial data:

* `detect_epsg(path)`: Reads the LAZ Variable Length Records (VLRs) to auto-detect the coordinate reference system.
* `write_csv(records, path)`: Dumps flat tabular data of tree metrics.
* `write_geojson(records, path)` & `write_gpkg(...)`: Generates OGC-compliant spatial vectors representing tree crowns, allowing seamless integration with QGIS or ArcGIS.
* `write_crown_polygons(...)`: Converts basic radii or convex hulls into Shapely polygon geometries for vector exports.
* `write_segments(...)`: Crops out individual segmented trees and streams them directly into `.laz` file.
* `write_report(...)`: Compiles a detailed summary text file outlining processing times, parameter choices, and a leaderboard of the tallest/widest trees extracted.

### `cli.py` & `models.py`

* `models.py:TreeRecord`: A dataclass enforcing a rigid schema for extracted metrics (Tree ID, X, Y, Z, Height, Crown Diameter, Crown Area, Volume, DBH, Point Count).
* `models.py:ExtractionParams`: Consolidates user configurations (resolution, window sizes, formats) into a single transportable object.
* `cli.py:main()`: Maps raw shell arguments into `ExtractionParams`, handles directory creation (`output_dir`), and configures the standard Python logger for intuitive terminal feedback.

---

## Algorithm Details

### 1. Ground Model (DTM)
When classified ground points (LAS class `2`) are available, they are collected globally during the initial streaming pass and rasterized into a shared 2m-resolution DTM. This DTM is shared across all multiprocessing worker processes, eliminating redundant ground model computation per tile and ensuring seamless terrain transitions at chunk borders.
When no ground classification exists, a vectorized Cloth Simulation Filter (CSF, Zhang et al. 2016) is deployed to classify the ground dynamically from scratch.

### 2. Tree Segmentation
1. **CHM Rasterization**: Non-ground points are rasterized into a 2D Canopy Height Model (CHM) at a configurable high resolution.
2. **Adaptive Smoothing**: Gaussian smoothing is applied to remove anomalous noise spikes, with the sigma ($\sigma$) scaling proportionally to the grid resolution.
3. **Variable-Window Maxima**: Treetop detection is initiated where the search radius for local maxima scales allometrically with local canopy height: `crown_radius = 0.1 × height + 1.0`.
4. **Watershed Segmentation**: The identified treetops act as seeds for a marker-controlled watershed algorithm, which floods the inverted CHM landscape to delineate precise contiguous crown boundaries.

### 3. Crown Metrics & Allometry
* **Feret Diameter**: Computed as the maximum distance between any two vertices of the upper canopy's 2D convex hull.
* **Crown Area**: The area of the 2D convex hull of the delineated canopy.
* **Crown Volume**: Approximated using a 3D convex hull with a bounding-box fallback for sparse canopies.
* **Diameter at Breast Height (DBH)**: Estimated using a generalized global allometric model based on tree height ($H$) and crown diameter ($CD$):
  `DBH = 1.2 × H^0.74 × CD^0.26` *(Jucker et al. 2017)*.

---

## Output Structure & Formats

For a sample input file named `cloud85.laz`, the script automatically creates a `cloud85/` subdirectory containing:

* **`trees.csv`** - Spreadsheet of extracted trees and their complete geometric attributes.
* **`trees.gpkg`** - GeoPackage containing geospatial points/polygons, ready for direct loading into QGIS/ArcGIS.
* **`report.txt`** - Text report summarizing processing time, extraction parameters, and top trees.
* **`segments.laz`** - *(optional, requires `--export-segments`)* a file containing individual extracted tree point cloud.

---
