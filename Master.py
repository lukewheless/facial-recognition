import face_recognition
import cvlib as cv
import cv2
import numpy as np
import tensorflow as tp

video_capture = cv2.VideoCapture(0)
padding = 20

dean_image = face_recognition.load_image_file("Dean.PNG")
dean_face_encoding = face_recognition.face_encodings(dean_image)[0]

sean_image = face_recognition.load_image_file("Sean.PNG")
sean_face_encoding = face_recognition.face_encodings(sean_image)[0]

drake_image = face_recognition.load_image_file("Drake.PNG")
drake_face_encoding = face_recognition.face_encodings(drake_image)[0]

fletch_image = face_recognition.load_image_file("Fletch.PNG")
fletch_face_encoding = face_recognition.face_encodings(fletch_image)[0]

luke_image = face_recognition.load_image_file("Luke.PNG")
luke_face_encoding = face_recognition.face_encodings(luke_image)[0]

ami_image = face_recognition.load_image_file("Ami.PNG")
ami_face_encoding = face_recognition.face_encodings(ami_image)[0]

josh_image = face_recognition.load_image_file("Josh.PNG")
josh_face_encoding = face_recognition.face_encodings(josh_image)[0]

jamy_image = face_recognition.load_image_file("Jamy.PNG")
jamy_face_encoding = face_recognition.face_encodings(jamy_image)[0]

amy_image = face_recognition.load_image_file("Amy.PNG")
amy_face_encoding = face_recognition.face_encodings(amy_image)[0]

avery_image = face_recognition.load_image_file("Avery.PNG")
avery_face_encoding = face_recognition.face_encodings(avery_image)[0]

hannah_image = face_recognition.load_image_file("Hannah.PNG")
hannah_face_encoding = face_recognition.face_encodings(hannah_image)[0]

burch_image = face_recognition.load_image_file("Burch.PNG")
burch_face_encoding = face_recognition.face_encodings(burch_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    dean_face_encoding,
    sean_face_encoding,
    drake_face_encoding,
    fletch_face_encoding,
    luke_face_encoding,
    ami_face_encoding,
    josh_face_encoding,
    jamy_face_encoding,
    amy_face_encoding,
    avery_face_encoding,
    hannah_face_encoding,
    burch_face_encoding
]

known_face_names = [
    "Dean",
    "Sean",
    "Drake",
    "Fletcher",
    "Luke",
    "Ami",
    "Josh",
    "Jamy",
    "Amy",
    "Avery",
    "Hannah",
    "Jared"
]
    
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face, confidence = cv.detect_face(frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    i = 0

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    for idx, f in enumerate(face):    
        i = i+1
        startX, startY = max(0, f[0]-padding), max(0, f[1]-padding)
        endX, endY = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
        cv2.rectangle(frame, (startX + 10, startY + 70), (endX - 10, endY - 20), (0,255,0), 2)
        cv2.rectangle(frame, (startX + 10, endY - 50), (endX - 10, endY - 20), (0, 0, 255), cv2.FILLED)

        face_crop = np.copy(frame[startY:endY, startX:endX]) 
        label, confidence = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        label = label[idx]
        label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
        Y = startY + 70 if startY - 70 > 70 else startY + 70
        
        cv2.putText(frame, label, (startX + 10, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, name, (startX + 12, endY - 25), 1, font, (255, 255, 255), 3) 
    cv2.putText(frame, 'Faces = ' + str(i), (50, 50), font, 1, (0, 255, 255), 2)
    cv2.imshow("Facial Recognition", frame)
    process_this_frame = not process_this_frame

    print("Total Number of Faces Detected:", len(face_locations))
    print("Gender is", label)

    # press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows() 
