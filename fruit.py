import cv2
from ultralytics import YOLO

class FruitDetector:
    def __init__(self, confidence_threshold=0.6):
        # Charger le modèle YOLOv8 pré-entraîné
        self.model = YOLO('yolov8n.pt')
        
        # Définir le seuil de confiance
        self.confidence_threshold = confidence_threshold

        # Mapping des classes pour les fruits courants
        self.fruit_classes = {
            47: 'Pomme',    # Apple
            52: 'Banane',   # Banana
            49: 'Orange',   # Orange
        }

        # Dictionnaire pour compter les fruits détectés
        self.fruit_counts = {fruit: 0 for fruit in self.fruit_classes.values()}

    def detect_fruits(self, frame):
        # Réinitialiser les compteurs avant chaque détection
        self.fruit_counts = {fruit: 0 for fruit in self.fruit_classes.values()}

        # Effectuer la détection des objets sur l'image (frame)
        results = self.model(frame)

        # Parcourir les résultats et détecter les fruits dans la liste des classes
        for result in results:
            for bbox, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                fruit_id = int(cls)

                # Vérifier si la confiance dépasse le seuil défini
                if fruit_id in self.fruit_classes and conf > self.confidence_threshold:
                    fruit_name = self.fruit_classes[fruit_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{fruit_name} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Incrémenter le compteur pour le type de fruit détecté
                    self.fruit_counts[fruit_name] += 1

        # Afficher les compteurs de fruits sur l'interface de la caméra
        self.display_fruit_counts(frame)

        return frame

    def display_fruit_counts(self, frame):
        # Positionner les compteurs de fruits sur l'interface de la caméra
        y_offset = 30
        for fruit, count in self.fruit_counts.items():
            cv2.putText(frame, f"{fruit}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_offset += 30

    def get_fruit_counts(self):
        # Retourner les comptages de fruits détectés
        return self.fruit_counts
