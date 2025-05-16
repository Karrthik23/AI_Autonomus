import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class YoloDeepSortNode(Node):
    def __init__(self):
        super().__init__('yolo_deepsort_node')

        self.model = YOLO('yolov8s.pt')  # Use yolov8n.pt if you want faster test

        self.tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)

        self.bridge = CvBridge()

        self.subscriber = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            10
        )

        self.get_logger().info("YOLOv8 + DeepSORT node started, waiting for images...")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLOv8 detection
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2, y2], conf, cls))

        # DeepSORT tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("YOLO + DeepSORT", frame)
        cv2.waitKey(1)  # Needed to refresh the imshow window

def main(args=None):
    rclpy.init(args=args)
    node = YoloDeepSortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
