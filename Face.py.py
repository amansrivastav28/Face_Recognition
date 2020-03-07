import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#Preparing Data Set to trained Machine
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_caputre(image):
    """
    This Faction take image as argument and convert into Gray Scale Layer
    and classifier detect and strore in faces variable and then  cropped face
    from image and store it in folder
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for(x, y, w, h) in faces:
        cropped_face = image[y:y+h, x:x+w]
    return cropped_face
#open camera and store in cap variable
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_caputre(frame) is not None:
        count += 1
        face = cv2.resize(face_caputre(frame), (200, 200))    #resize face image
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file ='image/'+str(count)+'.jpg'   #store in jpg format in image folder
        cv2.imwrite(file, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Cropped Face', face)
    else:
        pass
    if cv2.waitKey(1) == 13 or count == 20:
        cv2.putText(face, "Dataset Prepared", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Cropped Face', face)
        break
cap.release()      #releasing camera
cv2.destroyAllWindows()  #destroying all windows accquired by camera

#Start to Trained Model
data_path = 'image/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], [] #createing list

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   #raed image with grayscale
    Training_Data.append(np.asarray(images, dtype=np.uint8))  #append image in Training data
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

#Model Training complete Here

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)  #predict image through trainned model and store in result variable

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300)) #It find confidence percentage
            display_string = str(confidence)+'% Face'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if confidence > 75:
            cv2.putText(image, "Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Reconigition', image)   #Open Camera Window

        else:
            cv2.putText(image, "Not Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Reconigition', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Reconigition', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()
