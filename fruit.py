import cv2
from ultralytics import YOLO

class FruitDetector:
    def __init__(self):
        # Charger le modèle YOLOv8 pré-entraîné (coco128 contient certains fruits)
        self.model = YOLO('yolov8n.pt')  # Utilise un modèle YOLOv8 pré-entrainé

        # Mapping des classes pour les fruits courants
        self.fruit_classes = {
            47: 'Pomme',    # Apple
            52: 'Banane',   # Banana
            49: 'Orange',   # Orange
            # Si des classes spécifiques existent pour Poire, Kiwi, Avocat, Fraise, Framboise
            # elles devraient être ajoutées ici. Pour le moment, ce sont des exemples.
            # Tu devras potentiellement ré-entraîner un modèle sur ces classes.
            # Exemple hypothétique :
            # 53: 'Poire',  # Ajout hypothétique pour Poire
            # 54: 'Kiwi',   # Ajout hypothétique pour Kiwi
            # 55: 'Avocat', # Ajout hypothétique pour Avocat
            # 56: 'Fraise', # Ajout hypothétique pour Fraise
            # 57: 'Framboise' # Ajout hypothétique pour Framboise
        }

    def detect_fruits(self, frame):
        # Effectuer la détection des objets sur l'image (frame)
        results = self.model(frame)

        # Parcourir les résultats et détecter les fruits dans la liste des classes
        for result in results:
            for bbox, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                fruit_id = int(cls)
                if fruit_id in self.fruit_classes:
                    fruit_name = self.fruit_classes[fruit_id]
                    x1, y1, x2, y2 = map(int, bbox)  # Extraire les coordonnées de la boîte englobante
                    label = f"{fruit_name} ({conf:.2f})"  # Ajouter une étiquette avec la confiance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessiner le rectangle
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame
