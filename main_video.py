import cv2

from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.change_access_level()
sfr.load_encoding_images()

# Load Camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()


    # Detect Faces
    face_locations, face_names, assess_state = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if assess_state == 2:
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.putText(frame, "Access Granted!", (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
        elif assess_state == 1:
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)
            cv2.putText(frame, "Not high enough access level!", (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 4)
        else:
            cv2.putText(frame, "Unknown face!", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
