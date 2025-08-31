import cv2
import dlib
import numpy as np
import pygame.mixer
from scipy.spatial import distance

# Initialize pygame mixer for alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("beep-warning-6387.mp3")

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define eye landmarks for left and right eye
LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]

# State tracking variables
sleep = drowsy = active = 0
status = ""
color = (0, 0, 0)
alert_triggered = False
EYE_AR_THRESH_SLEEP = 0.20
EYE_AR_THRESH_DROWSY = 0.25
EYE_AR_CONSEC_FRAMES = 6

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        continue

    frame = cv2.flip(frame, 1)
    faces = detector(frame)

    for face in faces:
        landmarks = predictor(frame, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < EYE_AR_THRESH_SLEEP:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep >= EYE_AR_CONSEC_FRAMES:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if not alert_triggered:
                    alert_sound.play()
                    alert_triggered = True
        elif ear < EYE_AR_THRESH_DROWSY:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy >= EYE_AR_CONSEC_FRAMES:
                status = "Drowsy !"
                color = (0, 0, 255)
                if not alert_triggered:
                    alert_sound.play()
                    alert_triggered = True
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active >= EYE_AR_CONSEC_FRAMES:
                status = "Active :)"
                color = (0, 255, 0)
                alert_triggered = False

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
