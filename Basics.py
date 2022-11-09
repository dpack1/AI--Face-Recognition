import cv2
import numpy as np
import face_recognition

#loading images and converting them into rgb
imgPractice = face_recognition.load_image_file('ImagesBasic/Dylan.jpg')
imgPractice = cv2.cvtColor(imgPractice, cv2.COLOR_RGB2BGR)
imgTest = face_recognition.load_image_file('ImagesBasic/Couple.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)

#face location
faceLoc = face_recognition.face_locations(imgPractice)[0]
encodePractice = face_recognition.face_encodings(imgPractice)[0]
cv2.rectangle(imgPractice,(faceLoc[3], faceLoc[0],faceLoc[1], faceLoc[2]),(255,0,255),2)

print(faceLoc) #location of face (topright, bottom right, topleft, bottomleft)
print("Top Right: " + str(faceLoc[0]))
print("Bottom Right: " + str(faceLoc[1]))
print("Top Left: " + str(faceLoc[2]))
print("Bottom Left: " + str(faceLoc[3]))

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0],faceLocTest[1], faceLocTest[2]),(255,0,255),2)

#compare faces

results = face_recognition.compare_faces([encodePractice],encodeTest)
faceDis = face_recognition.face_distance([encodePractice], encodeTest)

print(results, faceDis)

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)



cv2.imshow('Practice', imgPractice)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)
