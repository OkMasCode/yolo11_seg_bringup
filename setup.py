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
    install_requires=['setuptools'],
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
            '3d_yolo11_seg_node2 = yolo11_seg_bringup.3d_yolo11_seg_node2:main',
            '3d_yolo11_seg_node_main = yolo11_seg_bringup.yolo11_seg_node_main:main',
            'stereo_to_pc_node = yolo11_seg_bringup.stereo_to_pc:main',
            'mapper_node2 = yolo11_seg_bringup.mapper_node2:main',
            'clip_reader = yolo11_seg_bringup.clip_reader:main',
        ],
    },
)
