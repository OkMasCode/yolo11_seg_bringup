# Code Cleanup and Synchronization Improvements

## Changes Applied

### 1. **3d_yolo11_seg_node2.py**

#### Removed Code Duplication
- ✅ Removed duplicate `CvBridge` import

#### Improved RGB/Depth Synchronization
- ✅ Added timestamp checking (50ms threshold) between RGB and Depth messages
- ✅ Added warning logs for timestamp mismatches
- ✅ Process messages outside of lock to avoid blocking callbacks
- ✅ This prevents processing mismatched frames that could cause artifacts

#### Enhanced DetectedObject Publishing
- ✅ Added validation to ensure centroid exists before publishing
- ✅ Graceful handling of missing embeddings (empty array if not found)
- ✅ Better error messages with instance IDs for debugging
- ✅ Consolidated message creation for clarity

### 2. **mapper_node2.py**

#### Removed Unused Imports
- ✅ Removed: `PointCloud2`, `point_cloud2`, `Point`, `YOLO`, `defaultdict`
- ✅ Kept only necessary imports for cleaner code

#### Added Missing Functionality
- ✅ Added marker publisher for visualization
- ✅ Implemented `publish_centroids()` method to visualize mapped objects
- ✅ Creates sphere markers and text labels for each detected object

---

## Additional Suggestions for Synchronization

### 1. **Use message_filters for Exact Time Synchronization**

Instead of manual synchronization, consider using ROS2 `message_filters`:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber

# In __init__:
self.rgb_sub = Subscriber(self, Image, self.image_topic)
self.depth_sub = Subscriber(self, Image, self.depth_topic)

self.sync = ApproximateTimeSynchronizer(
    [self.rgb_sub, self.depth_sub],
    queue_size=10,
    slop=0.05  # 50ms tolerance
)
self.sync.registerCallback(self.synced_cb)
```

**Benefits:**
- Automatic timestamp matching
- Handles message queue properly
- Reduces risk of processing mismatched frames

### 2. **Add Processing Rate Limiting**

Prevent overwhelming the system during high frame rates:

```python
# In __init__:
self.last_process_time = self.get_clock().now()
self.min_process_interval = 0.1  # Process at most 10 Hz

# In rgb_callback or synced_cb:
current_time = self.get_clock().now()
time_diff = (current_time - self.last_process_time).nanoseconds * 1e-9
if time_diff < self.min_process_interval:
    return
self.last_process_time = current_time
```

### 3. **Use Threading for Heavy Processing**

Move YOLO and CLIP inference to separate thread:

```python
import queue
from threading import Thread

# In __init__:
self.processing_queue = queue.Queue(maxsize=2)
self.processing_thread = Thread(target=self._processing_worker, daemon=True)
self.processing_thread.start()

def rgb_callback(self, msg):
    try:
        self.processing_queue.put_nowait((msg, self.latest_depth_msg))
    except queue.Full:
        self.get_logger().warn("Processing queue full, dropping frame")

def _processing_worker(self):
    while rclpy.ok():
        try:
            rgb_msg, depth_msg = self.processing_queue.get(timeout=1.0)
            self.synced_cb(rgb_msg, depth_msg)
        except queue.Empty:
            continue
```

**Benefits:**
- Callbacks return quickly
- No blocking of ROS2 spin
- Automatic frame dropping when processing is slow

### 4. **Add Detection Confidence Tracking**

Store embeddings with detection history for better mapping:

```python
# In mapper_node2.py custom_callback:
if msg.embedding:
    # Store embedding for similarity comparison with future detections
    embedding_np = np.array(msg.embedding, dtype=np.float32)
    
    # Optional: Use embedding similarity for better object association
    # Compare with existing objects using cosine similarity
```

### 5. **Implement Proper Shutdown Handling**

```python
# In 3d_yolo11_seg_node2.py:
def destroy_node(self):
    """Clean up resources before shutdown."""
    self.get_logger().info("Shutting down node...")
    # Clear GPU memory
    if hasattr(self, 'model'):
        del self.model
    if hasattr(self, 'model2'):
        del self.model2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    super().destroy_node()
```

### 6. **Add Data Validation**

```python
def validate_detection_data(self):
    """Ensure all detection data is consistent."""
    # Check all lists have same length or matching instance IDs
    meta_ids = {m["instance_id"] for m in self.last_detection_meta}
    centroid_ids = {c["instance_id"] for c in self.last_centroids}
    
    if meta_ids != centroid_ids:
        self.get_logger().warn(
            f"Mismatch in detection data: "
            f"meta={len(meta_ids)}, centroids={len(centroid_ids)}"
        )
        return False
    return True
```

### 7. **Memory Management for Embeddings**

```python
# Limit stored embeddings to prevent memory growth
MAX_STORED_EMBEDDINGS = 1000

# In mapper_node2.py:
class PointCloudMapperNode(Node):
    def __init__(self):
        # ...
        self.embedding_cache = {}  # {object_id: embedding}
        
    def custom_callback(self, msg):
        # Store embedding with object_id
        if msg.embedding and len(msg.embedding) > 0:
            if len(self.embedding_cache) >= MAX_STORED_EMBEDDINGS:
                # Remove oldest entries (FIFO)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[object_id] = np.array(msg.embedding)
```

### 8. **QoS Profile Optimization**

Consider using `RELIABLE` for DetectedObject messages:

```python
# In mapper_node2.py:
qos_reliable = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure message delivery
    durability=DurabilityPolicy.VOLATILE
)

self.cm_sub = self.create_subscription(
    DetectedObject, 
    self.cm_topic, 
    self.custom_callback, 
    qos_profile=qos_reliable
)
```

---

## Testing Recommendations

1. **Test timestamp synchronization:**
   ```bash
   ros2 topic echo /camera/camera/color/image_raw --field header.stamp &
   ros2 topic echo /camera/camera/aligned_depth_to_color/image_raw --field header.stamp
   ```

2. **Monitor processing rate:**
   ```bash
   ros2 topic hz /yolo/detections
   ```

3. **Check for dropped frames:**
   Monitor warning logs for "RGB/Depth timestamp mismatch" messages

4. **Verify embedding sizes:**
   ```bash
   ros2 topic echo /yolo/detections --field embedding --once
   # Should show 512 float values for ViT-B/32
   ```

---

## Summary

The main synchronization improvements focus on:
- **Thread safety**: Proper locking and processing outside locks
- **Timestamp validation**: Ensuring RGB/Depth alignment
- **Data consistency**: Validating all detection data matches
- **Graceful degradation**: Handling missing data appropriately
- **Clean code**: Removing duplicates and unused imports

These changes will make the system more robust and easier to debug.
