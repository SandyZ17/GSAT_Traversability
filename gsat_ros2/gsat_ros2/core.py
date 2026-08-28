"""ROS-independent helpers for point-cloud decoding and GSAT map generation."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


FLOAT32_POINT_FIELD = 7


def resolve_gsat_python_path(explicit_path: str = "") -> Path:
    """Find the directory that contains GSAT's top-level ``model`` package."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("GSAT_PYTHON_PATH"):
        candidates.append(Path(os.environ["GSAT_PYTHON_PATH"]).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "gsat",
            Path.cwd() / "src" / "GSAT_Traversability" / "gsat",
            Path.cwd() / "gsat",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "model" / "__init__.py").is_file():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate GSAT's Python directory. Set the ROS parameter "
        f"'gsat_python_path' or GSAT_PYTHON_PATH. Checked: {checked}"
    )


def import_gsat_models(explicit_path: str = "") -> tuple[Any, Any]:
    """Import the existing GSAT feature extractor and head without copying them."""
    gsat_path = resolve_gsat_python_path(explicit_path)
    if str(gsat_path) not in sys.path:
        sys.path.insert(0, str(gsat_path))
    model_module = importlib.import_module("model")
    return model_module.geo_feature, model_module.gsat_head


def _field_map(fields: Iterable[Any]) -> dict[str, Any]:
    return {field.name: field for field in fields}


def pointcloud2_to_xyzi(message: Any) -> np.ndarray:
    """Decode float32 x/y/z and optional intensity fields from PointCloud2.

    The decoder honors ``point_step`` and row padding. Non-finite XYZ samples are
    removed. Missing intensity values are filled with zero.
    """
    fields = _field_map(message.fields)
    missing = [name for name in ("x", "y", "z") if name not in fields]
    if missing:
        raise ValueError(f"PointCloud2 is missing required fields: {missing}")

    selected = [fields[name] for name in ("x", "y", "z")]
    if any(field.datatype != FLOAT32_POINT_FIELD for field in selected):
        raise ValueError("GSAT requires float32 x, y, and z PointCloud2 fields")

    intensity = fields.get("intensity")
    if intensity is not None and intensity.datatype != FLOAT32_POINT_FIELD:
        raise ValueError("GSAT requires a float32 intensity field when present")

    endian = ">" if message.is_bigendian else "<"
    dtype_fields = [
        ("x", f"{endian}f4", fields["x"].offset),
        ("y", f"{endian}f4", fields["y"].offset),
        ("z", f"{endian}f4", fields["z"].offset),
    ]
    if intensity is not None:
        dtype_fields.append(("intensity", f"{endian}f4", intensity.offset))

    dtype = np.dtype(
        {
            "names": [item[0] for item in dtype_fields],
            "formats": [item[1] for item in dtype_fields],
            "offsets": [item[2] for item in dtype_fields],
            "itemsize": message.point_step,
        }
    )

    rows = []
    raw = memoryview(message.data)
    packed_row_size = message.width * message.point_step
    for row_index in range(message.height):
        start = row_index * message.row_step
        row = np.frombuffer(raw[start:start + packed_row_size], dtype=dtype)
        rows.append(row)
    structured = np.concatenate(rows) if rows else np.empty((0,), dtype=dtype)

    points = np.zeros((structured.size, 4), dtype=np.float32)
    for index, name in enumerate(("x", "y", "z")):
        points[:, index] = structured[name]
    if intensity is not None:
        points[:, 3] = structured["intensity"]

    return points[np.isfinite(points[:, :3]).all(axis=1)]


def transform_points(points: np.ndarray, translation: Sequence[float],
                     quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Apply a rigid transform to XYZI points."""
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ValueError("Transform quaternion has zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
             2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
             2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
             1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    result = points.copy()
    result[:, :3] = points[:, :3] @ rotation.T + np.asarray(
        translation, dtype=np.float32
    )
    return result


def filter_roi(points: np.ndarray, point_range: Sequence[float]) -> np.ndarray:
    """Keep finite points inside an inclusive six-value XYZ range."""
    if len(point_range) != 6:
        raise ValueError("point_range must contain [x0, y0, z0, x1, y1, z1]")
    x0, y0, z0, x1, y1, z1 = point_range
    finite = np.isfinite(points[:, :3]).all(axis=1)
    mask = (
        finite
        & (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )
    return np.ascontiguousarray(points[mask], dtype=np.float32)


def observed_cell_mask(points: np.ndarray, point_range: Sequence[float],
                       shape: tuple[int, int]) -> np.ndarray:
    """Return an HxW mask for BEV cells containing at least one input point."""
    height, width = shape
    x0, y0, _, x1, y1, _ = point_range
    mask = np.zeros((height, width), dtype=bool)
    if points.size == 0:
        return mask
    grid_x = np.floor((points[:, 0] - x0) / ((x1 - x0) / width)).astype(int)
    grid_y = np.floor((points[:, 1] - y0) / ((y1 - y0) / height)).astype(int)
    valid = (
        (grid_x >= 0) & (grid_x < width)
        & (grid_y >= 0) & (grid_y < height)
    )
    mask[grid_y[valid], grid_x[valid]] = True
    return mask


def occupancy_values(distance_map: np.ndarray, threshold: float,
                     observed_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert anomaly distances to ROS OccupancyGrid values.

    Normal cells are free (0), anomalous cells are lethal (100), and cells with
    no LiDAR return are unknown (-1) when an observation mask is provided.
    """
    values = np.where(distance_map < threshold, 0, 100).astype(np.int8)
    if observed_mask is not None:
        if observed_mask.shape != distance_map.shape:
            raise ValueError("observed_mask and distance_map shapes must match")
        values[~observed_mask] = -1
    return values
