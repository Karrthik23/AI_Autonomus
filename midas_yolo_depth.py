import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load YOLOv8 model (person detection only)
yolo = YOLO("yolov8n.pt")  # use yolov8s.pt for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Running MiDaS + YOLOv8...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO to detect people
    results = yolo(frame)
    person_box = None
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_box = (x1, y1, x2, y2)
                break  # Take first detected person
        if person_box:
            break

    # Run MiDaS depth estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_tensor)
        depth_map = prediction.squeeze().cpu().numpy()

    # Normalize + colorize depth map
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = depth_map.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # Resize to match original frame
    depth_colormap = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # If person found, draw box and estimate depth
    if person_box:
        x1, y1, x2, y2 = person_box
        roi = depth_map[y1:y2, x1:x2]
        estimated_depth = np.mean(roi)

        # Draw box + depth
        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(depth_colormap, f"Depth: {estimated_depth:.1f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Combine views
    combined = np.hstack((frame, depth_colormap))
    cv2.imshow("YOLOv8 + MiDaS Depth Estimation", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
