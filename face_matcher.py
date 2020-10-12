import cv2
import face_recognition as fr
import os
import numpy as np

pathunknown = 'faces'
img = []
classNames = []
mylist = os.listdir(pathunknown)

for cl in mylist:
    curImg = cv2.imread(f'{pathunknown}/{cl}')
    img.append(curImg)
    classNames.append(os.pathunknown.splitext(cl)[0])

imageknown = fr.load_image_file('familiar_face/1_dw1.jpg')
imageknown = cv2.cvtColor(imageknown,cv2.COLOR_BGR2RGB)
encodeknown = fr.face_encodings(imageknown)[0]

def findEncodings(img):
    encodelist = []
    for photo in img:
        photo = cv2.cvtColor(photo , cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(photo)[0]
        encodelist.append(encode)

    return encodelist

encodeListKnownFaces = findEncodings(img)
print("Encoding has been completed ..")



for f in encodeListUnKnownFaces:
    matches = fr.compare_faces(encodeknown,f)
    if matches:
        cv2.imwrite(base_dir + 'facesfound/' +  'new' + file, f)
        break
        



