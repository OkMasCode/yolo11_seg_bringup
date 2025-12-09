#!/usr/bin/env python3
"""
Launch file for YOLO segmentation pipeline with semantic mapping and reader.

This launch file starts:
1. yolo11_seg_node_main - YOLO detection with CLIP embeddings
2. mapper_node2 - Semantic object mapping
3. clip_reader - Semantic map visualization/printer
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/sensor/yolov8n-seg.engine',
        description='Path to YOLO model file'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/camera/color/image_raw',
        description='RGB image topic'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/camera/aligned_depth_to_color/image_raw',
        description='Depth image topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera/color/camera_info',
        description='Camera info topic'
    )
    
    text_prompt_arg = DeclareLaunchArgument(
        'text_prompt',
        default_value='a photo of a person',
        description='CLIP text prompt for similarity'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='camera_color_optical_frame',
        description='Fixed frame for semantic map'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_color_optical_frame',
        description='Camera frame for detections'
    )
    
    # YOLO segmentation node
    yolo_node = Node(
        package='yolo11_seg_bringup',
        executable='3d_yolo11_seg_node_main',
        name='yolo11_seg_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'text_prompt': LaunchConfiguration('text_prompt'),
            'pointcloud_topic': '/yolo/pointcloud',
            'annotated_topic': '/yolo/annotated',
            'clip_boxes_topic': '/yolo/clip_boxes',
            'detections_topic': '/yolo/detections',
            'conf': 0.25,
            'iou': 0.70,
            'imgsz': 640,
            'retina_masks': True,
            'depth_scale': 1000.0,
            'pc_downsample': 2,
            'pc_max_range': 8.0,
            'mask_threshold': 0.5,
            'clip_square_scale': 1.4,
            'debug_clip_boxes': False,
            'publish_annotated': False,
            'publish_clip_boxes_vis': False,
        }]
    )
    
    # Semantic mapper node
    mapper_node = Node(
        package='yolo11_seg_bringup',
        executable='mapper_node',
        name='pointcloud_mapper_node',
        output='screen',
        parameters=[{
            'detection_message': '/yolo/detections',
            'output_dir': '/home/sensor/ros2_ws/src/yolo11_seg_bringup',
            'export_interval': 5.0,
            'map_frame': LaunchConfiguration('map_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'semantic_map_topic': '/yolo/semantic_map',
        }]
    )
    
    # Semantic map reader/printer node
    reader_node = Node(
        package='yolo11_seg_bringup',
        executable='clip_reader',
        name='semantic_map_printer',
        output='screen',
    )
    
    return LaunchDescription([
        model_path_arg,
        image_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        text_prompt_arg,
        map_frame_arg,
        camera_frame_arg,
        yolo_node,
        mapper_node,
        reader_node,
    ])
