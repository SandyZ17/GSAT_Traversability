# GSAT ROS 2

Native ROS 2 online inference wrapper for GSAT. The node subscribes to a
`sensor_msgs/msg/PointCloud2`, runs the existing PyTorch/CUDA model, and
publishes:

- `/gsat/traversability` (`nav_msgs/msg/OccupancyGrid`): `0` normal/free,
  `100` anomalous/lethal, and `-1` no LiDAR observation.
- `/gsat/anomaly_score` (`sensor_msgs/msg/Image`, `32FC1`): latent distance to
  the learned positive center.

The package does not retrain or copy the GSAT model. `point_range`,
`voxel_size`, and the checkpoint must come from the same training run.

See [PLAN.md](PLAN.md) for the cross-machine handoff checklist, pending runtime
validation, and Nav2 integration milestones.

## Requirements

- ROS 2 with `rclpy`, `tf2_ros`, `sensor_msgs`, and `nav_msgs`
- PyTorch compatible with the installed CUDA runtime
- Python packages required by `../gsat`
- The compiled GSAT voxelization extension

```bash
cd <ros2-workspace>/src/GSAT_Traversability/gsat
python3 setup.py build_ext --inplace

cd <ros2-workspace>
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select gsat_ros2
source install/setup.bash
```

Edit `config/inference.yaml`, especially `checkpoint_path` and
`gsat_python_path`, then run:

```bash
ros2 launch gsat_ros2 inference.launch.py
```

For a temporary override:

```bash
ros2 run gsat_ros2 inference_node --ros-args \
  -p checkpoint_path:=/data/checkpoints/best_model.pth \
  -p gsat_python_path:=/ws/src/GSAT_Traversability/gsat \
  -p input_topic:=/lidar/points
```

The same node accepts live LiDAR, simulation, or rosbag2 data. Its subscription
uses the sensor-data QoS profile:

```bash
ros2 bag play <bag-directory>
ros2 topic echo /gsat/traversability --once
```

## Frames and map convention

By default, incoming points are transformed into `base_link` using TF. Set
`target_frame` to an empty string only when the cloud is already in the desired
output frame. The OccupancyGrid origin is the configured `(x_min, y_min)`, and
its resolution is derived from the model output width and `point_range`.

This is a robot-centric local grid, not a persistent world map. To consume it in
Nav2, implement a Costmap2D layer that copies/transforms these costs at each
update; do not feed it to a static-map server as though it were globally fixed.

## Diagnostics

```bash
ros2 topic info /points_raw --verbose
ros2 topic hz /gsat/traversability
ros2 run tf2_ros tf2_echo base_link <lidar_frame>
```

If the model cannot be imported, verify that `gsat_python_path` points to the
directory containing `model/` and `ops/`, and that `ops/voxel_op*.so` exists.
