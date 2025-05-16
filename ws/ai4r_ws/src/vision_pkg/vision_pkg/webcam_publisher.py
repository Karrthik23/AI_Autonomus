import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher = self.create_publisher(Image, '/image', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 0 = default webcam

        if not self.cap.isOpened():
            self.get_logger().error("Webcam not found!")
        else:
            self.get_logger().info("Webcam stream started.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

