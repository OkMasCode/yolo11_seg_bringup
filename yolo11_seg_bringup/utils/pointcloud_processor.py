from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField

import torch
import numpy as np
import struct

class PointCloudProcessor:
    """ 
    Handles PointCloud generation from depth and segmentation masks. 
    """
    def __init__(self, fx, fy, cx, cy, device, depth_scale = 1000.0, 
                 pc_downsample = 2, pc_max_range = 8.0):
         
        self.depth_scale = depth_scale
        self.pc_downsample = pc_downsample
        self.pc_max_range = pc_max_range
        self.device = device

        self.fx_t = torch.tensor(fx, dtype=torch.float32, device=device)
        self.fy_t = torch.tensor(fy, dtype=torch.float32, device=device)
        self.cx_t = torch.tensor(cx, dtype=torch.float32, device=device)
        self.cy_t = torch.tensor(cy, dtype=torch.float32, device=device)

    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> float:
        """ 
        Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field. 
        """
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]
    
    @staticmethod
    def publish_pointcloud(points: np.ndarray, header, publisher):
        """
        Publish PointCloud2 message from points numpy array.
        """
        try:
            if points is None or points.size == 0:
                return

            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32),
                ('class_id', np.int32),
                ('instance_id', np.int32),
            ])

            structured_points = np.zeros(points.shape[0], dtype=dtype)
            structured_points['x'] = points[:, 0].astype(np.float32)
            structured_points['y'] = points[:, 1].astype(np.float32)
            structured_points['z'] = points[:, 2].astype(np.float32)
            structured_points['rgb'] = points[:, 3].astype(np.float32)
            structured_points['class_id'] = points[:, 4].astype(np.int32)
            structured_points['instance_id'] = points[:, 5].astype(np.int32)

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='class_id', offset=16, datatype=PointField.INT32, count=1),
                PointField(name='instance_id', offset=20, datatype=PointField.INT32, count=1),
            ]
            cloud_msg = point_cloud2.create_cloud(header, fields, structured_points)
            publisher.publish(cloud_msg)
        except Exception as e:
            print(f"Error publishing pointcloud: {e}")

    def process_detection(self, mask_t, depth_t, valid_mask_t, class_id, instance_id, 
                         rgb_color, scale_factor, min_points=10):
        # 1. Apply Mask
        obj_mask_t = (mask_t).to(self.device)
        valid_t = valid_mask_t & obj_mask_t

        v_coords_t, u_coords_t = valid_t.nonzero(as_tuple=True)
        if v_coords_t.shape[0] < min_points:
            return None, None
        
        z_vals_t = (depth_t[v_coords_t, u_coords_t].mul_(scale_factor))

        # 2. HISTOGRAM FILTERING (Fixes the Drift)
        # Instead of averaging everything (Mean), find the DENSEST depth (Mode)
        # Create 50 bins from 0 to Max Range (e.g. 0 to 8m)
        hist = torch.histc(z_vals_t, bins=50, min=0, max=self.pc_max_range)
        max_bin_idx = torch.argmax(hist) 
        
        # Calculate the depth of the peak bin
        bin_width = self.pc_max_range / 50.0
        peak_depth = max_bin_idx * bin_width + (bin_width / 2.0)
        
        # Only keep points within 20cm of the peak (cuts off the floor at 8m)
        keep_mask_t = (z_vals_t >= peak_depth - 0.2) & (z_vals_t <= peak_depth + 0.2)
        
        if not torch.any(keep_mask_t):
            return None, None
        
        z_clean_t = z_vals_t[keep_mask_t]
        u_clean_t = u_coords_t[keep_mask_t].float()
        v_clean_t = v_coords_t[keep_mask_t].float()

        # ... (Downsampling code same as before) ...
        if self.pc_downsample and self.pc_downsample > 1:
            step_t = int(self.pc_downsample)
            idx = torch.arange(0, z_clean_t.shape[0], step_t, device=self.device, dtype=torch.long)
            z_clean_t = z_clean_t[idx]
            u_clean_t = u_clean_t[idx]
            v_clean_t = v_clean_t[idx]

        x_t = (u_clean_t - self.cx_t) * z_clean_t / self.fx_t
        y_t = (v_clean_t - self.cy_t) * z_clean_t / self.fy_t

        # 3. Use MEDIAN for Centroid (Stable against outliers)
        centroid = (
            float(torch.median(x_t).item()),       # Median X
            float(torch.median(y_t).item()),       # Median Y
            float(torch.median(z_clean_t).item())  # Median Z
        )

        # ... (Rest of the packing code same as before) ...
        N = x_t.shape[0]
        r, g, b = rgb_color
        rgb_packed = self.pack_rgb(r, g, b)
        instance_cloud_t = torch.stack([x_t, y_t, z_clean_t, torch.full((N,), float(rgb_packed), dtype=torch.float32, device=self.device), torch.full((N,), int(class_id), dtype=torch.int32, device=self.device), torch.full((N,), int(instance_id), dtype=torch.int32, device=self.device)], dim=1)
        
        return instance_cloud_t, centroid

    def prepare_depth_tensor(self, depth_img, encoding, scale_factor):
        """ 
        Convert depth image to GPU tensor with validity mask. 
        """
        depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(self.device)
        valid_mask_t = (depth_t > 0) & (~torch.isnan(depth_t))
        if self.pc_max_range > 0.0:
            valid_mask_t = valid_mask_t & (depth_t * scale_factor <= self.pc_max_range)
        return depth_t, valid_mask_t