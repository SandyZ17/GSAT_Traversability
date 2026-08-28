"""Runtime wrapper around the existing GSAT PyTorch model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .core import filter_roi, import_gsat_models


class GsatRuntime:
    """Load a GSAT checkpoint once and infer anomaly distances from XYZI points."""

    def __init__(self, checkpoint_path: str, point_range: Sequence[float],
                 voxel_size: Sequence[float], device_name: str = "auto",
                 gsat_python_path: str = "") -> None:
        import torch

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GSAT checkpoint not found: {checkpoint}")

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested, but CUDA is unavailable")

        self.torch = torch
        self.device = torch.device(device_name)
        self.point_range = [float(value) for value in point_range]
        self.voxel_size = [float(value) for value in voxel_size]

        geo_feature, gsat_head = import_gsat_models(gsat_python_path)
        payload = torch.load(str(checkpoint), map_location=self.device)
        self.feature_model = geo_feature(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_range,
        ).to(self.device)
        self.feature_model.load_state_dict(payload["feature_model"])
        self.feature_model.eval()

        self.head_model = gsat_head().to(self.device)
        self.head_model.load_state_dict(payload["head_model"])
        self.head_model.eval()

        center = payload.get("center_c")
        if center is None:
            raise KeyError("Checkpoint does not contain required 'center_c'")
        self.center = center.reshape(1, -1).to(self.device)

        radius = payload.get("mean_radius_val")
        if radius is None:
            raise KeyError("Checkpoint does not contain required 'mean_radius_val'")
        self.radius = float(radius.item() if hasattr(radius, "item") else radius)

    def infer(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distance_map, filtered_points)`` for one cloud."""
        filtered = filter_roi(points, self.point_range)
        if filtered.size == 0:
            raise ValueError("Point cloud contains no points inside the configured ROI")

        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(filtered).to(self.device)
            _, _, feature_map = self.feature_model([tensor])
            latent, _, _, _ = self.head_model(feature_map)
            _, channels, height, width = latent.shape
            flattened = latent.permute(0, 2, 3, 1).reshape(-1, channels)
            center = self.center.to(dtype=flattened.dtype)
            distance = self.torch.sqrt(
                self.torch.sum((flattened - center) ** 2, dim=1).clamp(min=0.0)
                + 1e-6
            )
            distance_map = distance.reshape(height, width).cpu().numpy()
        return distance_map.astype(np.float32, copy=False), filtered
