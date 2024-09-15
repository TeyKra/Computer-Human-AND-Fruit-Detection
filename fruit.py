import cv2
from ultralytics import YOLO

class FruitDetector:
    def __init__(self, confidence_threshold=0.6):
        # Load the pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        # Set the confidence threshold for fruit detection
        self.confidence_threshold = confidence_threshold

        # Mapping for common fruit classes (ID to label)
        self.fruit_classes = {
            47: 'Apple',    # Apple
            49: 'Orange'    # Orange
        }

        # Dictionary to count the detected fruits
        self.fruit_counts = {fruit: 0 for fruit in self.fruit_classes.values()}

    def detect_fruits(self, frame):
        # Reset fruit counters before each detection
        self.fruit_counts = {fruit: 0 for fruit in self.fruit_classes.values()}

        # Perform object detection on the image (frame)
        results = self.model(frame)

        # Iterate through results and detect fruits based on their class ID
        for result in results:
            for bbox, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                fruit_id = int(cls)

                # Check if the confidence exceeds the defined threshold
                if fruit_id in self.fruit_classes and conf > self.confidence_threshold:
                    fruit_name = self.fruit_classes[fruit_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{fruit_name} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Increment the count for the detected fruit type
                    self.fruit_counts[fruit_name] += 1

        # Display the fruit counts on the camera interface
        self.display_fruit_counts(frame)

        return frame

    def display_fruit_counts(self, frame):
        # Display fruit counts on the camera interface
        y_offset = 30
        for fruit, count in self.fruit_counts.items():
            cv2.putText(frame, f"{fruit}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_offset += 30

    def get_fruit_counts(self):
        # Return the counts of detected fruits
        return self.fruit_counts
