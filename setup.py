from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo11_seg_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'scikit-learn'],
    zip_safe=True,
    maintainer='sensor',
    maintainer_email='sensor@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pc_vision_node_v3 = yolo11_seg_bringup.pc_vision_v3:main',
            'map_points_node = yolo11_seg_bringup.map_points_node:main',
            'clustered_map_points_node = yolo11_seg_bringup.clustered_map_points_node:main',
            'cpp_mapper_json_exporter_node = yolo11_seg_bringup.cpp_mapper_json_exporter_node:main',
            'cluster_assignment_node = yolo11_seg_bringup.cluster_assignment_node:main',
        ],
    },
)
