import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class SimpleYoloAnnotatorNode(Node):
    def __init__(self):
        super().__init__('simple_yolo_annotator_node')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/vision/annotated_image')
        self.declare_parameter('model_path', '/workspaces/yoloe-26m-seg.pt')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('iou', 0.45)

        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_sensor,
        )
        self.annotated_pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f'Publishing annotated images on: {self.output_topic}')

    def image_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.model.set_classes(["microwave"])
            results = self.model.predict(
                source=cv_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                stream=False,
            )

            annotated = results[0].plot() if results else cv_bgr

            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.annotated_pub.publish(out_msg)

        except Exception as exc:
            self.get_logger().error(f'Failed to process image: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleYoloAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
