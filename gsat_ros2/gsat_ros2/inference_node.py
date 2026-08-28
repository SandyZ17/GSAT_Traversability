"""ROS 2 node that publishes GSAT traversability maps from PointCloud2."""

from __future__ import annotations

from array import array
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .core import (
    observed_cell_mask,
    occupancy_values,
    pointcloud2_to_xyzi,
    transform_points,
)
from .model_runtime import GsatRuntime


class GsatInferenceNode(Node):
    """Run GSAT on live, simulated, or rosbag2 PointCloud2 messages."""

    def __init__(self) -> None:
        super().__init__("gsat_inference")
        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("occupancy_topic", "/gsat/traversability")
        self.declare_parameter("score_topic", "/gsat/anomaly_score")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("gsat_python_path", "")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("transform_timeout_sec", 0.1)
        self.declare_parameter("point_range", [-9.0, -9.0, -2.0, 9.0, 9.0, 2.0])
        self.declare_parameter("voxel_size", [0.15, 0.15, 4.0])
        self.declare_parameter("threshold_scale", 1.0)
        self.declare_parameter("device", "auto")
        self.declare_parameter("min_points", 100)
        self.declare_parameter("unknown_for_empty", True)
        self.declare_parameter("publish_score_image", True)
        self.declare_parameter("log_period_sec", 2.0)

        self.point_range = list(
            self.get_parameter("point_range").get_parameter_value().double_array_value
        )
        self.voxel_size = list(
            self.get_parameter("voxel_size").get_parameter_value().double_array_value
        )
        if len(self.point_range) != 6 or len(self.voxel_size) != 3:
            raise ValueError("point_range must have 6 values and voxel_size must have 3")
        if any(value <= 0.0 for value in self.voxel_size):
            raise ValueError("voxel_size values must be positive")
        if any(
            self.point_range[index] >= self.point_range[index + 3]
            for index in range(3)
        ):
            raise ValueError("point_range minima must be smaller than maxima")

        checkpoint_path = self.get_parameter("checkpoint_path").value
        gsat_python_path = self.get_parameter("gsat_python_path").value
        self.runtime = GsatRuntime(
            checkpoint_path=checkpoint_path,
            point_range=self.point_range,
            voxel_size=self.voxel_size,
            device_name=self.get_parameter("device").value,
            gsat_python_path=gsat_python_path,
        )
        self.threshold = (
            self.runtime.radius * float(self.get_parameter("threshold_scale").value)
        )
        if self.threshold <= 0.0:
            raise ValueError("The effective anomaly threshold must be positive")
        self.target_frame = self.get_parameter("target_frame").value
        self.transform_timeout = float(
            self.get_parameter("transform_timeout_sec").value
        )
        self.min_points = int(self.get_parameter("min_points").value)
        self.unknown_for_empty = bool(self.get_parameter("unknown_for_empty").value)
        self.publish_score_image = bool(
            self.get_parameter("publish_score_image").value
        )
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)
        self.last_log_time = 0.0
        self.processing_lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_publisher = self.create_publisher(
            OccupancyGrid, self.get_parameter("occupancy_topic").value, 1
        )
        self.score_publisher = self.create_publisher(
            Image, self.get_parameter("score_topic").value, 1
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter("input_topic").value,
            self._cloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"GSAT ready on {self.runtime.device}; threshold={self.threshold:.4f}; "
            f"input={self.get_parameter('input_topic').value}"
        )

    def _lookup_transform(self, message: PointCloud2) -> TransformStamped | None:
        source_frame = message.header.frame_id
        if not self.target_frame or source_frame == self.target_frame:
            return None
        return self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame,
            Time.from_msg(message.header.stamp),
            timeout=Duration(seconds=self.transform_timeout),
        )

    @staticmethod
    def _apply_transform(points: np.ndarray,
                         transform: TransformStamped | None) -> np.ndarray:
        if transform is None:
            return points
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_points(
            points,
            (translation.x, translation.y, translation.z),
            (rotation.x, rotation.y, rotation.z, rotation.w),
        )

    def _cloud_callback(self, message: PointCloud2) -> None:
        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().debug("Dropping cloud because inference is still running")
            return
        try:
            started = time.perf_counter()
            points = pointcloud2_to_xyzi(message)
            transform = self._lookup_transform(message)
            points = self._apply_transform(points, transform)
            if points.shape[0] < self.min_points:
                self.get_logger().warning(
                    f"Skipping sparse cloud with {points.shape[0]} points"
                )
                return

            distance_map, filtered = self.runtime.infer(points)
            observed = None
            if self.unknown_for_empty:
                observed = observed_cell_mask(
                    filtered, self.point_range, distance_map.shape
                )
            costs = occupancy_values(distance_map, self.threshold, observed)
            output_frame = self.target_frame or message.header.frame_id
            self.map_publisher.publish(
                self._make_occupancy_grid(costs, message.header.stamp, output_frame)
            )
            if self.publish_score_image:
                self.score_publisher.publish(
                    self._make_score_image(
                        distance_map, message.header.stamp, output_frame
                    )
                )

            now = time.monotonic()
            if now - self.last_log_time >= self.log_period_sec:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.get_logger().info(
                    f"Published {distance_map.shape[1]}x{distance_map.shape[0]} "
                    f"map from {filtered.shape[0]} ROI points in {elapsed_ms:.1f} ms"
                )
                self.last_log_time = now
        except TransformException as error:
            self.get_logger().warning(f"TF lookup failed: {error}")
        except (ValueError, RuntimeError) as error:
            self.get_logger().error(f"GSAT inference skipped: {error}")
        finally:
            self.processing_lock.release()

    def _make_occupancy_grid(self, costs: np.ndarray, stamp: TimeMessage,
                             frame_id: str) -> OccupancyGrid:
        height, width = costs.shape
        x0, y0, _, x1, y1, _ = self.point_range
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        resolution_x = (x1 - x0) / width
        resolution_y = (y1 - y0) / height
        if not np.isclose(resolution_x, resolution_y):
            raise ValueError(
                "OccupancyGrid requires equal X/Y resolution, but the model "
                f"produced {resolution_x} and {resolution_y}"
            )
        message.info.map_load_time = stamp
        message.info.resolution = float(resolution_x)
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = float(x0)
        message.info.origin.position.y = float(y0)
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        message.data = array("b", costs.reshape(-1).tolist())
        return message

    @staticmethod
    def _make_score_image(scores: np.ndarray, stamp: TimeMessage,
                          frame_id: str) -> Image:
        contiguous = np.ascontiguousarray(scores, dtype="<f4")
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height, message.width = contiguous.shape
        message.encoding = "32FC1"
        message.is_bigendian = False
        message.step = message.width * contiguous.dtype.itemsize
        message.data = contiguous.tobytes()
        return message


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GsatInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
