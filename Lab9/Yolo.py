import cv2
from ultralytics import YOLO



model = YOLO("yolov8n.pt")   # you can change model

video_path = "video.mp4"   

cap = cv2.VideoCapture("Yolovid.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()


while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml"
    )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Press ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()