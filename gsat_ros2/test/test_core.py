from dataclasses import dataclass
import unittest

import numpy as np

from gsat_ros2.core import (
    filter_roi,
    observed_cell_mask,
    occupancy_values,
    pointcloud2_to_xyzi,
    transform_points,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int = 7


@dataclass
class Cloud:
    fields: list
    data: bytes
    width: int
    height: int
    point_step: int
    row_step: int
    is_bigendian: bool = False


def make_cloud(points, include_intensity=True):
    width = len(points)
    point_step = 16 if include_intensity else 12
    names = ["x", "y", "z"] + (["intensity"] if include_intensity else [])
    fields = [Field(name, index * 4) for index, name in enumerate(names)]
    packed = np.asarray(points, dtype="<f4")[:, :len(names)].tobytes()
    return Cloud(fields, packed, width, 1, point_step, width * point_step)


class CoreTest(unittest.TestCase):

    def test_pointcloud2_to_xyzi_preserves_intensity_and_drops_nan(self):
        cloud = make_cloud([[1.0, 2.0, 3.0, 4.0], [np.nan, 0.0, 0.0, 5.0]])
        result = pointcloud2_to_xyzi(cloud)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0, 4.0]])

    def test_pointcloud2_to_xyzi_fills_missing_intensity(self):
        cloud = make_cloud([[1.0, 2.0, 3.0, 99.0]], include_intensity=False)
        result = pointcloud2_to_xyzi(cloud)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0, 0.0]])

    def test_pointcloud2_requires_float_xyz(self):
        cloud = make_cloud([[1.0, 2.0, 3.0, 4.0]])
        cloud.fields[0].datatype = 5
        with self.assertRaisesRegex(ValueError, "float32"):
            pointcloud2_to_xyzi(cloud)

    def test_transform_points_applies_rotation_and_translation(self):
        points = np.array([[1.0, 0.0, 0.0, 7.0]], dtype=np.float32)
        half = np.sqrt(0.5)
        result = transform_points(
            points, [1.0, 2.0, 0.0], [0.0, 0.0, half, half]
        )
        np.testing.assert_allclose(result, [[1.0, 3.0, 0.0, 7.0]], atol=1e-6)

    def test_roi_mask_and_occupancy_values(self):
        points = np.array(
            [
                [-0.5, -0.5, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        point_range = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        filtered = filter_roi(points, point_range)
        self.assertEqual(filtered.shape, (2, 4))

        observed = observed_cell_mask(filtered, point_range, (2, 2))
        distances = np.array([[0.2, 2.0], [2.0, 0.2]], dtype=np.float32)
        costs = occupancy_values(distances, threshold=1.0, observed_mask=observed)
        np.testing.assert_array_equal(costs, [[0, -1], [-1, 0]])


if __name__ == "__main__":
    unittest.main()
