import cv2
from human import HumanDetector
from fruit import FruitDetector

def main():
    # Créer une instance des détecteurs
    human_detector = HumanDetector()
    fruit_detector = FruitDetector()

    # Afficher un menu dans le terminal
    print("Sélectionnez le mode de détection :")
    print("1. Détection des humains")
    print("2. Détection des fruits")
    print("3. Détection des humains et des fruits")
    choix = input("Entrez votre choix (1/2/3) : ")

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Appliquer les détections en fonction du choix de l'utilisateur
        if choix == '1':
            frame = human_detector.process_frame(frame)
        elif choix == '2':
            frame = fruit_detector.detect_fruits(frame)
        elif choix == '3':
            frame = human_detector.process_frame(frame)
            frame = fruit_detector.detect_fruits(frame)
        else:
            print("Choix invalide. Veuillez relancer le programme et sélectionner une option valide.")
            break

        # Afficher le frame annoté
        cv2.imshow("Détection en cours", frame)

        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
