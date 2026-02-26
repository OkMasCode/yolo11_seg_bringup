import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
# 1. IMPORT QoS MODULES
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class CloudInspector(Node):
    def __init__(self):
        super().__init__('cloud_inspector')

        # 2. CREATE A "BEST EFFORT" QOS PROFILE
        # This matches what RealSense and most cameras use
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.listener_callback,
            qos_profile=qos_profile) # 3. PASS THE PROFILE HERE

    def listener_callback(self, msg):
        print("\n--- New Message ---")
        print(f"Height: {msg.height}")
        print(f"Width:  {msg.width}")
        print(f"Is Dense: {msg.is_dense}")
        print(f"Point Step: {msg.point_step} bytes")
        print(f"Row Step:   {msg.row_step} bytes")
        print(f"Data Length: {len(msg.data)} bytes")
        
        expected_size = msg.height * msg.width * msg.point_step
        print(f"Expected Size (H*W*Step): {expected_size}")
        
        if len(msg.data) != expected_size:
            diff = len(msg.data) - expected_size
            print(f"MISMATCH: Data is {diff} bytes {'larger' if diff > 0 else 'smaller'} than expected.")
        else:
            print("MATCH: Data size is perfect.")
            
        # Exit after one message
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    inspector = CloudInspector()
    try:
        rclpy.spin(inspector)
    except SystemExit:
        pass
    inspector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()