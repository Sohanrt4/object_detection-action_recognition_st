from action_model import predict_action
import cv2
import argparse
import os
from ultralytics import YOLO

# COCO class IDs we want to detect
# person = 0, bicycle = 1, car = 2, motorcycle = 3, bus = 5, truck = 7
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect_objects(input_path, output_path):
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # lightweight and fast

    # Open input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Please wait.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection on the frame
        results = model.track(frame, persist=True, verbose=False)

        # Process detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else -1

                # Only keep person and vehicles
                if class_id == 0 and confidence > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    action = predict_action(frame)
                    label = f"ID {track_id} - {TARGET_CLASSES[class_id]} - {action}"

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        (0, 255, 0),
                        -1
                    )

                    # Put label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection in Video using YOLOv8")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output_detected.mp4", help="Path to save output video")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
    else:
        detect_objects(args.input, args.output)