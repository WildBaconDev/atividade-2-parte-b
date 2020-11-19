import face_recognition
import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

# Removi a foto para subir no git kkk
known_image = face_recognition.load_image_file("Davi.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
class_name = "Davi"

while True:
      
   ret, frame = cap.read()
   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   try:
      face_locations = face_recognition.face_locations(frame_rgb, model="hog")
      matches = []
      unknown_encodings_list = face_recognition.face_encodings(frame_rgb, face_locations)

      for unknown_encoding in unknown_encodings_list:
         results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)
         matches.append(results[0])
   except Exception as e:
      print(e)

   for face_tuple, match in zip(face_locations, matches):
      cv2.rectangle(frame, (face_tuple[3], face_tuple[0]), (face_tuple[1], face_tuple[2]), (0, 255, 0), 2)
      cv2.rectangle(frame, (face_tuple[3], face_tuple[2]-35), (face_tuple[1], face_tuple[2]), (0, 255, 0), cv2.FILLED)

      font = cv2.FONT_HERSHEY_DUPLEX

      if match:
         cv2.putText(frame, class_name, (face_tuple[3] + 6, face_tuple[2] - 6), font, 1.0, (255, 255, 255), 1)
      else:
         cv2.putText(frame, "DESCONHECIDO", (face_tuple[3] + 6, face_tuple[2] - 6), font, 1.0, (255, 255, 255), 1)

   cv2.imshow('frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
