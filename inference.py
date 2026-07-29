from ultralytics import YOLO
import cv2
import time
 
MODEL_PATH = "runs/road_line_seg/weights/best.pt"   
VIDEO_PATH = "test.mp4"                            
OUTPUT_PATH = "output.mp4"

CONFIDENCE = 0.4

 
model = YOLO(MODEL_PATH)

 
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Could not open video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Inference
    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()
    fps_live = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps_live:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    writer.write(annotated_frame)

    #cv2.imshow("YOLO Inference", annotated_frame)

    #if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved to {OUTPUT_PATH}")