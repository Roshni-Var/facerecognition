import face_recognition
import cv2
import numpy as np 
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Load known faces
rosh_image = face_recognition.load_image_file("faces/rosh.jpg")
rajvir_image = face_recognition.load_image_file("faces/rajvir.jpg")


width_rajvir = rajvir_image.shape[1]
height_rajvir = rajvir_image.shape[0]

# print(face_recognition.face_encodings(rajvir_image, known_face_locations=[(0, width, height, 0)]))
# print(len(face_recognition.face_encodings(rajvir_image, known_face_locations=[(0, width, height, 0)])))

width_rosh = rosh_image.shape[1]
height_rosh = rosh_image.shape[0]

# print(face_recognition.face_encodings(rosh_image, known_face_locations=[(0, width, height, 0)]))
# print(len(face_recognition.face_encodings(rosh_image, known_face_locations=[(0, width, height, 0)])))

rosh_encoding = face_recognition.face_encodings(rosh_image)[0]
rajvir_encoding = face_recognition.face_encodings(rajvir_image, known_face_locations=[(0, width_rajvir, height_rajvir, 0)])[0]

known_face_encodings = [rosh_encoding, rajvir_encoding]
known_faces_names = ["Rosh", "Rajvir"]

#list of expected students
students = known_faces_names.copy()

face_locations = []
face_encodings = []

#Get the current date and time

now = datetime.now()
current_date =now.strftime("%D-%m-%y").replace("/", "-")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    #RECOGNIZE FACES
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encodings in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        print("matched", matches)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encodings)
        best_match_index = np.argmin(face_distance)
        
        name = ""
        
        if (matches[best_match_index]):
            name = known_faces_names[best_match_index]
            print("name -> ", name)
        #add the text if a person is present
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font,fontScale, fontColor, thickness,lineType )
        
        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M%S")
            lnwriter.writerow([name, current_time])
            
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()            
    