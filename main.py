import cv2
import mediapipe as mp
import numpy as np  # Ajout de numpy

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

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

def draw_body_contour(image, pose_landmarks, width, height):
    # Contours du torse
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

    # Fermer le contour en reconnectant les hanches et les épaules
    body_contour.append(body_contour[0])

    # Dessiner le contour du corps (torse)
    cv2.polylines(image, [np.array(body_contour)], isClosed=True, color=(255, 255, 255), thickness=2)

    # Contours des bras
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

    # Contours des jambes
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

while cap.isOpened():
    ret, frame = cap.read()

    # Convertir l'image en format RGB (MediaPipe utilise RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Obtenir les landmarks du corps, du visage et des mains
    result_pose = pose.process(rgb_frame)
    result_hands = hands.process(rgb_frame)
    result_face = face_mesh.process(rgb_frame)

    height, width, _ = frame.shape

    # Dessiner les landmarks du corps si détectés
    if result_pose.pose_landmarks:
        # Contours du corps
        draw_body_contour(frame, result_pose.pose_landmarks, width, height)
        
        # Dessiner les points du corps avec des connexions spécifiques
        mp_drawing.draw_landmarks(
            frame,
            result_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Style pour les points clés
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)   # Style pour les connexions
        )

    # Dessiner les landmarks des mains si détectés
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Style pour les points clés
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Style pour les connexions
            )

    # Dessiner les landmarks du visage si détectés
    if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:
            # Dessiner les landmarks du visage avec tesselation et contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Dessiner les contours des yeux, lèvres et visage
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # Dessiner les contours des yeux
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    # Afficher l'image avec les contours du corps, des mains et du visage
    cv2.imshow("Body, Hands, and Face Mesh Estimation", frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
