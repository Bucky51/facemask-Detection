# live_detection.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import datetime

# Load the trained model
mymodel = tf.keras.models.load_model('mymodel1.h5')

# Initialize OpenCV's face detection (using Haar Cascade)
face_cascade = cv2.CascadeClassifier('C:/Users/hp/OneDrive/Desktop/Proj1/FaceMaskDetector/haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

# Start processing video frames
while cap.isOpened():
    _, img = cap.read()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Crop the face region
        face_img = img[y:y+h, x:x+w]
        
        # Save the cropped face image temporarily
        cv2.imwrite('temp.jpg', face_img)
        
        # Load the image and prepare it for prediction
        test_image = image.load_img('temp.jpg', target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make a prediction
        prediction = mymodel.predict(test_image)[0][0]
        
        # Display results on the frame
        if prediction == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', (x + w // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', (x + w // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Add timestamp
        timestamp = str(datetime.datetime.now())
        cv2.putText(img, timestamp, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show the live video feed
    cv2.imshow('Face Mask Detection', img)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
