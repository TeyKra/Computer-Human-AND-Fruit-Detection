import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe Pose, Hands et Face Mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Fonction pour tracer une ligne verticale sur le cou (des épaules au menton)
def draw_vertical_line_neck(image, pose_landmarks, face_landmarks, width, height):
    # Calculer la position moyenne des épaules
    x_left_shoulder = int(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    y_left_shoulder = int(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    x_right_shoulder = int(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    y_right_shoulder = int(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    
    # Position moyenne des épaules
    x_neck = (x_left_shoulder + x_right_shoulder) // 2
    y_neck = (y_left_shoulder + y_right_shoulder) // 2

    # Calculer la position du menton
    x_chin = int(face_landmarks.landmark[152].x * width)  # Le landmark 152 correspond au menton
    y_chin = int(face_landmarks.landmark[152].y * height)
    
    # Tracer la ligne verticale du cou
    cv2.line(image, (x_neck, y_neck), (x_chin, y_chin), (0, 255, 0), 2)  # Ligne verte pour le cou

# Fonction pour annoter les épaules, cou et pectoraux
def annotate_shoulders_neck_pectoral(image, pose_landmarks, width, height):
    body_parts = {
        "epaule gauche": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "epaule droite": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "coup": None,
        "pectoral gauche": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
        "pectoral droit": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP]
    }

    for part, landmarks in body_parts.items():
        if part == "coup":
            x_left = int(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
            y_left = int(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
            x_right = int(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
            y_right = int(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
            x = (x_left + x_right) // 2
            y = (y_left + y_right) // 2 - int(0.1 * height)
        elif isinstance(landmarks, list):
            x_shoulder = int(pose_landmarks.landmark[landmarks[0]].x * width)
            y_shoulder = int(pose_landmarks.landmark[landmarks[0]].y * height)
            x_hip = int(pose_landmarks.landmark[landmarks[1]].x * width)
            y_hip = int(pose_landmarks.landmark[landmarks[1]].y * height)
            x = (x_shoulder + x_hip) // 2
            y = (y_shoulder + y_hip) // 2 - int(0.15 * height)
        else:
            x = int(pose_landmarks.landmark[landmarks].x * width)
            y = int(pose_landmarks.landmark[landmarks].y * height)
        
        cv2.putText(image, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Fonction pour annoter les bras et avant-bras
def annotate_arms(image, pose_landmarks, width, height):
    body_parts = {
        "avant-bras gauche": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
        "avant-bras droit": [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
        "coude gauche": [mp_pose.PoseLandmark.LEFT_ELBOW],
        "coude droit": [mp_pose.PoseLandmark.RIGHT_ELBOW],
        "bras gauche": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
        "bras droit": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW]
    }

    for part, landmarks in body_parts.items():
        if len(landmarks) == 2:
            x1 = int(pose_landmarks.landmark[landmarks[0]].x * width)
            y1 = int(pose_landmarks.landmark[landmarks[0]].y * height)
            x2 = int(pose_landmarks.landmark[landmarks[1]].x * width)
            y2 = int(pose_landmarks.landmark[landmarks[1]].y * height)
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
        else:
            x = int(pose_landmarks.landmark[landmarks[0]].x * width)
            y = int(pose_landmarks.landmark[landmarks[0]].y * height)
        
        cv2.putText(image, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Fonction pour annoter les doigts, paumes, poignets et phalanges
def annotate_fingers_and_hands(image, hand_landmarks, handedness, width, height):
    finger_names = {
        4: "pouce",
        8: "index",
        12: "majeur",
        16: "annulaire",
        20: "auriculaire"
    }

    for idx, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        
        if idx in finger_names:
            finger_label = f"{finger_names[idx]} {handedness.lower()}"
            cv2.putText(image, finger_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if idx == 0:
            wrist_label = f"poignet {handedness.lower()}"
            cv2.putText(image, wrist_label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    palm_landmarks = [0, 5, 9, 13, 17]
    palm_x = int(sum([hand_landmarks.landmark[i].x for i in palm_landmarks]) / len(palm_landmarks) * width)
    palm_y = int(sum([hand_landmarks.landmark[i].y for i in palm_landmarks]) / len(palm_landmarks) * height)

    palm_label = f"paume {handedness.lower()}"
    cv2.putText(image, palm_label, (palm_x, palm_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    phalanges = {
        "1ere phalange": [5, 9, 13, 17],
        "2eme phalange": [6, 10, 14, 18],
        "3eme phalange": [7, 11, 15, 19]
    }

    for phalange_name, phalange_landmarks in phalanges.items():
        for i in phalange_landmarks:
            x = int(hand_landmarks.landmark[i].x * width)
            y = int(hand_landmarks.landmark[i].y * height)
            phalange_label = f"{phalange_name} {handedness.lower()}"
            cv2.putText(image, phalange_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Fonction pour annoter les parties du visage
def annotate_face_parts(image, face_landmarks, width, height):
    face_parts = {
        "oeil gauche": 33,
        "oeil droit": 263,
        "nez": 1,
        "joue gauche": 234,
        "joue droite": 454,
        "bouche": 13,
        "menton": 152,
        "front": 10,
        "sourcil gauche": 70,
        "sourcil droit": 300
    }

    for part, landmark_idx in face_parts.items():
        x = int(face_landmarks.landmark[landmark_idx].x * width)
        y = int(face_landmarks.landmark[landmark_idx].y * height)
        cv2.putText(image, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Fonction pour dessiner le contour du corps
def draw_body_contour(image, pose_landmarks, width, height):
    body_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    body_contour = []
    for point in body_points:
        x = int(pose_landmarks.landmark[point].x * width)
        y = int(pose_landmarks.landmark[point].y * height)
        body_contour.append((x, y))

    body_contour.append(body_contour[0])
    cv2.polylines(image, [np.array(body_contour)], isClosed=True, color=(255, 255, 255), thickness=2)

    arm_points_left = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST
    ]
    arm_points_right = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    
    for arm_points in [arm_points_left, arm_points_right]:
        arm_contour = []
        for point in arm_points:
            x = int(pose_landmarks.landmark[point].x * width)
            y = int(pose_landmarks.landmark[point].y * height)
            arm_contour.append((x, y))
        cv2.polylines(image, [np.array(arm_contour)], isClosed=False, color=(255, 255, 255), thickness=2)

    leg_points_left = [
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]
    leg_points_right = [
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    for leg_points in [leg_points_left, leg_points_right]:
        leg_contour = []
        for point in leg_points:
            x = int(pose_landmarks.landmark[point].x * width)
            y = int(pose_landmarks.landmark[point].y * height)
            leg_contour.append((x, y))
        cv2.polylines(image, [np.array(leg_contour)], isClosed=False, color=(255, 255, 255), thickness=2)

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Convertir l'image en format RGB (MediaPipe utilise RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Obtenir les landmarks du corps, du visage et des mains
    result_pose = pose.process(rgb_frame)
    result_hands = hands.process(rgb_frame)
    result_face = face_mesh.process(rgb_frame)

    height, width, _ = frame.shape

    if result_pose.pose_landmarks and result_face.multi_face_landmarks:
        draw_body_contour(frame, result_pose.pose_landmarks, width, height)
        
        mp_drawing.draw_landmarks(
            frame,
            result_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)
        )

        annotate_arms(frame, result_pose.pose_landmarks, width, height)
        annotate_shoulders_neck_pectoral(frame, result_pose.pose_landmarks, width, height)
        
        # Tracer la ligne verticale du cou entre les épaules et le menton
        for face_landmarks in result_face.multi_face_landmarks:
            draw_vertical_line_neck(frame, result_pose.pose_landmarks, face_landmarks, width, height)

    # Vérifier les landmarks des mains et dessiner si trouvés
    if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
        for hand_landmarks, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            handedness_label = handedness.classification[0].label
            annotate_fingers_and_hands(frame, hand_landmarks, handedness_label, width, height)

    if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:
            annotate_face_parts(frame, face_landmarks, width, height)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    # Afficher l'image avec toutes les annotations et la ligne verticale sur le cou
    cv2.imshow("Body, Hands, and Face Mesh Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
