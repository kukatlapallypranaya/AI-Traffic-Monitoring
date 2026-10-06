import cv2
from ultralytics import YOLO

# Load YOLO AI model
model = YOLO("yolo11n.pt")

# Open traffic video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Video could not be opened.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video completed.")
        break

    # Detect objects using AI
    results = model(frame, verbose=False)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Count vehicles
    vehicle_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in ["car", "bus", "truck", "motorcycle"]:
            vehicle_count += 1

    # Display vehicle count
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display video
    cv2.imshow(
        "AI Traffic Monitoring System",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
