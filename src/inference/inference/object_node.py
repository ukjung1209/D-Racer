import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from inference_msgs.msg import DetectionArray


class ObjectNode(Node):
    """카메라 이미지에서 신호등/표지판을 검출해 /object/detections로 발행한다."""

    def __init__(self):
        super().__init__('object_node')

        self.declare_parameter('image_topic', 'camera/image/compressed')
        self.declare_parameter('detection_topic', 'object/detections')

        image_topic = str(self.get_parameter('image_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)

        self.subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            10,
        )
        self.publisher = self.create_publisher(DetectionArray, detection_topic, 10)

        self.get_logger().info(
            f'object_node started: image_topic={image_topic}, '
            f'detection_topic={detection_topic}'
        )

    def image_callback(self, msg: CompressedImage):
        # TODO: onnxruntime YOLO 추론 -> Detection 리스트 채우기
        detections = DetectionArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera'
        detections.detections = []
        self.publisher.publish(detections)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
