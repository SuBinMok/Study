import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    roi = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:


        landmarks = predictor(gray, face)
        for i in range(0, 68):
            image =  cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,255,0), -1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()