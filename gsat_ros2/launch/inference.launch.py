"""Launch the GSAT ROS 2 online inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("gsat_ros2"), "config", "inference.yaml"
    )
    config_argument = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to the GSAT ROS 2 inference parameter file",
    )
    node = Node(
        package="gsat_ros2",
        executable="inference_node",
        name="gsat_inference",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_argument, node])
