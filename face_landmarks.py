import cv2
import numpy as np
import dlib
from PIL import Image
import datetime
import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def calculate(now):
    x1 = landmarks.part(43).x
    y1 = landmarks.part(43).y
    x2 = landmarks.part(40).x
    y2 = landmarks.part(40).y
    d1 = dist(x1, y1, x2, y2)
    d2 = dist(x1, y2, x2, y2)
    cv2.line(roi, (x1, y1), (x2, y2), (255,0,0))
    cv2.line(roi, (x1, y2), (x2, y2), (255, 255, 0))
    r = math.degrees(math.acos(d2 / d1))
    if r > 10:
        cv2.imwrite(".\image\img_"+ str(now) +".png", frame)
        alignment(r, now)

def alignment(r, now):
    image = cv2.imread(".\image\img_"+ str(now) +".png")
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), r, 1)
    print(matrix)
    # r_image = cv2.warpAffine((image, matrix, (rows, cols)))
    # cv2.imwrite(".\image\ro_"+ str(now) +".png", r_image)

def roi(image):
    w = abs((landmarks.part(0).x) - (landmarks.part(17).x))
    h = abs((landmarks.part(19).y) - (landmarks.part(8).y))
    x = w/2
    y = h/2
    return w,h,x,y
def dist(x1, y1, x2, y2):
    d = np.sqrt((x1- x2)**2 + (y1 - y2)**2)
    return d



while True:
    _, frame = cap.read()
    roi = cv2.flip(frame, 1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    for face in faces:
        landmarks = predictor(gray, face)
    for i in range(0, 68):
        image = cv2.circle(roi, (landmarks.part(i).x, landmarks.part(i).y), 1, (0,255,0), -1)
    image = cv2.circle(roi, (landmarks.part(42).x, landmarks.part(42).y), 1, (0, 0, 255), -1)#right
    image = cv2.circle(roi, (landmarks.part(39).x, landmarks.part(39).y), 1, (0, 0, 255), -1)#left
    calculate(now)


    cv2.imshow("roi", roi)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()