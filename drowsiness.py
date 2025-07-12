import cv2
import mediapipe as mp
import numpy as np

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices for MediaPipe Face Mesh (left and right eye)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
closed_eyes_frame_count = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                closed_eyes_frame_count += 1
            else:
                closed_eyes_frame_count = 0

            if closed_eyes_frame_count >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
