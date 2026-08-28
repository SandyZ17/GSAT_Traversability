from glob import glob
import os

from setuptools import find_packages, setup


package_name = "gsat_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GSAT maintainers",
    maintainer_email="d22g66@naver.com",
    description="ROS 2 online inference wrapper for GSAT traversability estimation.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "inference_node = gsat_ros2.inference_node:main",
        ],
    },
)
