from dataclasses import dataclass

@dataclass
class TreeRecord:
    tree_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z_ground: float = 0.0
    height_m: float = 0.0
    crown_diam_m: float = 0.0
    crown_area_m2: float = 0.0
    crown_volume_m3: float = 0.0
    dbh_cm: float = 0.0
    point_count: int = 0

@dataclass
class ExtractionParams:
    min_height: float = 0.5
    max_height: float = 10.0
    auto: bool = False
    chm_resolution: float = 0.5
    min_points: int = 15
    use_all: bool = False
    classes: tuple[int, ...] = (3, 4, 5)
    workers: int = 0
    subsample: float = 0.0

    chunk_size: float = 200.0
    chunk_buffer: float = 5.0
    nms_dist: float = 2.0

    ignore_classes: bool = False
    export_segments: bool = False
