# GSAT ROS 2 Development Plan

This document is the handoff checklist for continuing `gsat_ros2` on another
machine. Work from the `develop` branch. Do not treat the package as deployment
ready until Milestone 1 passes with a real checkpoint and PointCloud2 bag.

## Current State

Implemented:

- ROS 2 `ament_python` package and launch/config files.
- Sensor-data QoS subscription to `sensor_msgs/msg/PointCloud2`.
- Optional TF conversion into `target_frame`.
- Reuse of the existing GSAT PyTorch model and CUDA voxelization extension.
- Robot-centric `nav_msgs/msg/OccupancyGrid` output (`0` free, `100` anomaly,
  `-1` unobserved) and a `32FC1` anomaly-score image.
- ROS-independent tests for point decoding, transforms, ROI filtering, and map
  encoding.

Validation completed here: Python syntax compilation, package XML parsing,
setuptools metadata check, and `git diff --check`. The current host had no ROS 2,
`colcon`, NumPy, checkpoint, or rosbag2 data, so runtime and numerical tests are
still pending.

## Milestone 0 — Recreate the Environment

Use one supported ROS 2 distribution consistently. Confirm that the installed
PyTorch build, CUDA toolkit, compiler, and Python version are ABI-compatible.

```bash
git switch develop
cd <ros2-ws>/src/GSAT_Traversability/gsat
python3 setup.py build_ext --inplace
test -n "$(find ops -name 'voxel_op*.so' -print -quit)"

cd <ros2-ws>
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select gsat_ros2
source install/setup.bash
colcon test --packages-select gsat_ros2
colcon test-result --verbose
```

Inputs not stored in Git must be copied separately:

- a GSAT checkpoint containing `feature_model`, `head_model`, `center_c`, and
  `mean_radius_val`;
- a rosbag2 with PointCloud2 and the required TF tree;
- the exact `point_range` and `voxel_size` used during training.

## Milestone 1 — Rosbag2 Online-Inference Smoke Test

1. Update `config/inference.yaml` paths and topic/frame names.
2. Start `ros2 launch gsat_ros2 inference.launch.py`.
3. Play the bag with `ros2 bag play <bag-directory> --clock` when simulated time
   is required.
4. Confirm output dimensions, frame, timestamp, rate, and cost values.

```bash
ros2 topic info /points_raw --verbose
ros2 run tf2_ros tf2_echo base_link <lidar-frame>
ros2 topic hz /gsat/traversability
ros2 topic echo /gsat/traversability --once
ros2 topic echo /gsat/anomaly_score --once
```

Acceptance criteria:

- no checkpoint, CUDA-extension, QoS, or TF errors;
- output grid dimensions and resolution match the trained model;
- cells without LiDAR returns remain `-1`, not free;
- thresholded ROS 2 output matches offline `anomaly_inference.py` on the same
  cloud within floating-point tolerance;
- sustained inference does not accumulate stale point-cloud callbacks.

## Milestone 2 — Visualization and Regression Tests

- Add an RViz2 configuration for OccupancyGrid and anomaly score inspection.
- Add a fixture generated from one known `.bin` frame and expected offline
  distance map; store only small, redistributable test data.
- Test organized clouds, row padding, missing intensity, NaNs, TF rotation, empty
  ROI, incorrect checkpoint metadata, and CPU/CUDA device selection.
- Record latency distribution and GPU memory after warm-up, not only mean FPS.

## Milestone 3 — Nav2 Integration

Implement a separate Nav2 Costmap2D layer. It should consume the robot-centric
grid, transform cells at the grid timestamp, expire stale maps, and define the
policy for unknown/anomalous cells. Keep this separate from the inference node.

Acceptance criteria: planner queries use fresh data, stale grids become unknown,
and disabling the layer restores the original Nav2 behavior.

## Milestone 4 — Deployment Hardening

- Add a ROS 2 CUDA Docker image and compose service.
- Replace path-based model discovery with an installed Python package or explicit
  model artifact location.
- Add diagnostic status for inference latency, dropped frames, TF failures, and
  checkpoint metadata.
- Decide and add the repository license; `package.xml` currently says `TODO`.
- Test live LiDAR before changing safety-critical planner costs.

## Optional Milestone 5 — Data Collection Port

Only after inference is stable, port the ROS 1 `sw_sync` and `data_collection`
nodes to `rclcpp`. Preserve timestamp pairing and validate generated CSV/`.bin`
files against the ROS 1 implementation. This is not required for ROS 2 inference.

## Known Risks

- The current checkpoint radius is a global mean, not a calibrated safety bound.
- The map is local in `base_link`; it is not a persistent global occupancy map.
- Model imports depend on `gsat_python_path`, and the compiled extension must
  match the target Python/PyTorch/CUDA environment.
- ROS 2 runtime behavior remains unverified until Milestone 1 is completed.
