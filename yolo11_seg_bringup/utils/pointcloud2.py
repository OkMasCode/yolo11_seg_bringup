#!/usr/bin/env python3
"""Pointcloud generation utilities for YOLO segmentation."""
import struct
import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class PointCloudProcessor:
    """Handles pointcloud generation from depth and segmentation masks."""
    
    def __init__(self, fx, fy, cx, cy, device, depth_scale=1000.0, 
                 pc_downsample=2, pc_max_range=8.0, mask_threshold=0.5):
        """
        Initialize pointcloud processor.
        
        Args:
            fx, fy, cx, cy: Camera intrinsic parameters
            device: torch device (cuda/cpu)
            depth_scale: Depth scale factor (mm to meters)
            pc_downsample: Downsampling factor for pointcloud
            pc_max_range: Maximum depth range in meters
            mask_threshold: Threshold for binary mask
        """
        self.depth_scale = depth_scale
        self.pc_downsample = pc_downsample
        self.pc_max_range = pc_max_range
        self.mask_threshold = mask_threshold
        self.device = device
        
        # Cache intrinsic tensors
        self.fx_t = torch.tensor(fx, dtype=torch.float32, device=device)
        self.fy_t = torch.tensor(fy, dtype=torch.float32, device=device)
        self.cx_t = torch.tensor(cx, dtype=torch.float32, device=device)
        self.cy_t = torch.tensor(cy, dtype=torch.float32, device=device)
    
    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> float:
        """
        Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field.
        
        Args:
            r: Red channel (0-255)
            g: Green channel (0-255)
            b: Blue channel (0-255)
        Returns:
            float32 representing packed RGB value
        """
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]
    
    def process_detection(self, mask_t, depth_t, valid_mask_t, class_id, instance_id, 
                         rgb_color, scale_factor, min_points=10):
        """
        Process a single detection to generate pointcloud segment.
        
        Args:
            mask_t: Binary mask tensor (H, W)
            depth_t: Depth image tensor (H, W)
            valid_mask_t: Valid depth mask (H, W)
            class_id: Object class ID
            instance_id: Object instance ID
            rgb_color: Tuple (r, g, b)
            scale_factor: Depth to meters conversion factor
            min_points: Minimum points required
            
        Returns:
            tuple: (pointcloud_tensor, centroid_tuple) or (None, None) if insufficient points
        """
        obj_mask_t = (mask_t >= self.mask_threshold).to(self.device)
        valid_t = valid_mask_t & obj_mask_t
        
        v_coords_t, u_coords_t = valid_t.nonzero(as_tuple=True)
        if v_coords_t.numel() < min_points:
            return None, None
        
        z_vals_t = (depth_t[v_coords_t, u_coords_t] * scale_factor).to(torch.float32)
        
        # Fast outlier rejection using percentiles
        z_min = torch.quantile(z_vals_t, 0.1)
        z_max = torch.quantile(z_vals_t, 0.9)
        keep_mask_t = (z_vals_t >= z_min) & (z_vals_t <= z_max)
        if not torch.any(keep_mask_t):
            return None, None
        
        z_clean_t = z_vals_t[keep_mask_t]
        u_clean_t = u_coords_t[keep_mask_t].to(torch.float32)
        v_clean_t = v_coords_t[keep_mask_t].to(torch.float32)
        
        # Optional downsampling
        if self.pc_downsample and self.pc_downsample > 1:
            step_t = int(self.pc_downsample)
            idx = torch.arange(0, z_clean_t.shape[0], step_t, device=self.device)
            z_clean_t = z_clean_t[idx]
            u_clean_t = u_clean_t[idx]
            v_clean_t = v_clean_t[idx]
        
        # Convert to 3D coordinates
        x_clean_t = (u_clean_t - self.cx_t) * z_clean_t / self.fx_t
        y_clean_t = (v_clean_t - self.cy_t) * z_clean_t / self.fy_t
        
        # Compute centroid on GPU
        centroid_x = float(torch.mean(x_clean_t).item())
        centroid_y = float(torch.mean(y_clean_t).item())
        centroid_z = float(torch.mean(z_clean_t).item())
        
        # Build pointcloud tensor
        N = x_clean_t.shape[0]
        if N == 0:
            return None, None
        
        r, g, b = rgb_color
        rgb_packed = self.pack_rgb(r, g, b)
        
        rgb_packed_t = torch.full((N,), float(rgb_packed), dtype=torch.float32, device=self.device)
        class_id_t = torch.full((N,), float(class_id), dtype=torch.float32, device=self.device)
        instance_id_t = torch.full((N,), float(instance_id), dtype=torch.float32, device=self.device)
        
        instance_cloud_t = torch.stack(
            [
                x_clean_t.to(torch.float32),
                y_clean_t.to(torch.float32),
                z_clean_t.to(torch.float32),
                rgb_packed_t,
                class_id_t,
                instance_id_t,
            ],
            dim=1,
        )
        
        return instance_cloud_t, (centroid_x, centroid_y, centroid_z)
    
    def prepare_depth_tensor(self, depth_img, encoding, scale_factor):
        """
        Convert depth image to GPU tensor with validity mask.
        
        Args:
            depth_img: Numpy depth image
            encoding: ROS image encoding
            scale_factor: Depth conversion factor
            
        Returns:
            tuple: (depth_tensor, valid_mask_tensor)
        """
        depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(self.device)
        valid_mask_t = (depth_t > 0) & (~torch.isnan(depth_t))
        if self.pc_max_range > 0.0:
            valid_mask_t = valid_mask_t & (depth_t * scale_factor <= self.pc_max_range)
        return depth_t, valid_mask_t
    
    @staticmethod
    def publish_pointcloud(points: np.ndarray, header, publisher):
        """
        Publish PointCloud2 message.
        
        Args:
            points: Nx6 numpy array [x,y,z,rgb,class_id,instance_id]
            header: ROS message header
            publisher: ROS publisher
        """
        try:
            if points is None or points.size == 0:
                return
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="class_id", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="instance_id", offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            cloud_msg = point_cloud2.create_cloud(header, fields, points.tolist())
            publisher.publish(cloud_msg)
        except Exception as e:
            print(f"PointCloud publish failed: {e}")
