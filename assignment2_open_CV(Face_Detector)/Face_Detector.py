"""
This script captures video from the webcam and uses dlib's face detector to detect faces in real-time.
Author: Reza Toosi
"""

import cv2
import dlib

# Initialize dlib's face detector & Start capturing video
detector = dlib.get_frontal_face_detector()
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    # Detect faces in the frame
    faces = detector(frame)
    cv2.imshow('frame', frame)
    i = 0

    # Loop through the detected faces
    for face in faces:
        # Get the coordinates
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        i = i + 1

        # Put labels
        cv2.putText(frame, 'face ' + str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releases
vid.release()
cv2.destroyAllWindows()
