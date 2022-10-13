import os

import cv2
import face_recognition
import numpy as np

KEYCODE_ESC = 27
frame_resizing = 0.25
image_directory = "main_images"
unknown_name = "Unknown"
known_faces = []
known_names = []
capture = cv2.VideoCapture(0)

for file in os.listdir(image_directory):
    known_faces.append(face_recognition.face_encodings(
        cv2.cvtColor(cv2.imread(f"{image_directory}/{file}"), cv2.COLOR_BGR2RGB))[0])
    known_names.append(os.path.splitext(
        file.removeprefix(image_directory + "/").removesuffix("."))[0].replace("_", " "))

while True:
    names = []
    success, frame = capture.read()
    resized_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(resized_frame)
    face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

    for face in face_encodings:
        name = unknown_name
        best_match_index = np.argmin(face_recognition.face_distance(known_faces, face))
        if face_recognition.compare_faces(known_faces, face)[best_match_index]:
            name = known_names[best_match_index]
        names.append(name)

    for face_location, name in zip((np.array(face_locations) / frame_resizing).astype(int), names):
        y1, x2, y2, x1 = face_location[0], face_location[1], face_location[2], face_location[3]
        color = (0, 0, 200)
        if name != unknown_name:
            color = (0, 200, 0)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    if key == KEYCODE_ESC:
        break

capture.release()
cv2.destroyAllWindows()
