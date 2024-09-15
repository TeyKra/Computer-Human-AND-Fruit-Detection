import cv2
from human import HumanDetector
from fruit import FruitDetector
def main():
    # Create instances of detectors
    human_detector = HumanDetector()
    fruit_detector = FruitDetector()

    # Display a menu in the terminal
    print("Select detection mode:")
    print("1. Human detection")
    print("2. Fruit detection")
    print("3. Human and fruit detection")
    choice = input("Enter your choice (1/2/3): ")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Apply detections based on the user's choice
        if choice == '1':
            frame = human_detector.process_frame(frame)
        elif choice == '2':
            frame = fruit_detector.detect_fruits(frame)
        elif choice == '3':
            frame = human_detector.process_frame(frame)
            frame = fruit_detector.detect_fruits(frame)
        else:
            print("Invalid choice. Please restart the program and select a valid option.")
            break

        # Display the annotated frame
        cv2.imshow("Detection in progress", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
