import cv2
import numpy as np
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def facedata(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for(x, y, w, h) in faces:
        cropped = img[y:y+h, x:x+w]
    return cropped
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if facedata(frame) is not None:
        count += 1
        face = cv2.resize(facedata(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file ='image/'+str(count)+'.jpg'
        cv2.imwrite(file, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Cropper', face)
    else:
        print("Face Not Found")
        pass
    if cv2.waitKey(1) == 13 or count == 20:
        break
cap.release()
cv2.destroyAllWindows()
