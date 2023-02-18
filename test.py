import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    #cv2.imshow('img', img)
    # Stop if escape key is pressed
    #waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed. 
    # It takes time in milliseconds as a parameter and waits for the given time to destroy the window, 
    # if 0 is passed in the argument it waits till any key is pressed. 
    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()