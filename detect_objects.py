from action_model import predict_action
import cv2
import argparse
import os
from ultralytics import YOLO


TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect_objects(input_path, output_path):
 
    model = YOLO("yolov8n.pt")  

   
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {input_path}")
        return

  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

   
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Please wait.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

    
        results = model.track(frame, persist=True, verbose=False)

        person_centers = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else -1

               
                if class_id == 0 and confidence > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    person_centers.append((track_id, center_x, center_y))

                    action = predict_action(frame)
                    label = f"ID {track_id} - {TARGET_CLASSES[class_id]} - {action}"

                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                   
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

                  
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
        
        for i in range(len(person_centers)):
            for j in range(i + 1, len(person_centers)):

                id1, x1c, y1c = person_centers[i]
                id2, x2c, y2c = person_centers[j]

                distance = ((x1c - x2c) ** 2 + (y1c - y2c) ** 2) ** 0.5

                if distance < 60:
                    cv2.putText(
                        frame,
                        f"Tailgating Alert: {id1} & {id2}",
                        (50, 50 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
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
