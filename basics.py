# import necessary libraries
import cv2
import face_recognition

# load pictures
img_elon = face_recognition.load_image_file("basics_images/Elon_Musk.png")
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file("basics_images/Elon_Musk_2.png")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# detect location of the faces and draw rectangles around them
face_loc = face_recognition.face_locations(img_elon)[0]
encode_elon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (255, 0, 255), 2)

# compare the faces to check if they are from the same person
results = face_recognition.compare_faces([encode_elon], encode_test)

# calculate distance (the lower, the more equal)
face_dis = face_recognition.face_distance([encode_elon], encode_test)

# print results on the test picture
# the color either is green or red, depending on whether the faces are from the same person or not
color = (0, 0, 255)
if results == [True]:
    color = (0, 255, 0)
cv2.putText(img_test, f"{results} {round(face_dis[0], 2)}", (10, 45), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

# show pictures
cv2.imshow("Elon Musk", img_elon)
cv2.imshow("Test", img_test)
cv2.waitKey(0)
