import cv2
from human import HumanDetector
from fruit import FruitDetector

def main():
    # Initialiser le détecteur humain et le détecteur de fruits
    human_detector = HumanDetector()
    fruit_detector = FruitDetector()

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Traiter le frame pour les annotations humaines
        frame = human_detector.process_frame(frame)

        # Traiter le frame pour la détection de fruits
        frame = fruit_detector.detect_fruits(frame)

        # Afficher le frame annoté
        cv2.imshow("Body, Hands, Face, and Fruit Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
