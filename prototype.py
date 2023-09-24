#program hanya bisa dijalankan di python versi 3.7-3.10 (versi yang support tensorflow)
#bisa menggunakan anaconda jika versi python tidak support
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import img_to_array

# load model
json_file = open('model.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)

while True:
    check, test_img = vid.read()
    if check==False:
        continue
    cv2.putText(test_img, 'Press "q" to quit program', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    # turn image to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 6)

    for (x, y, w, h) in faces:
        # face image cropping
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)
        gray = gray_img[y:y+w, x:x+h]
        gray=cv2.resize(gray, (48, 48))
        
        # image data processing
        img_pixels=np.asarray(gray)
        img_pixels=img_pixels.astype('float')/255.0
        img_pixels=img_to_array(img_pixels)
        img_pixels=np.expand_dims(img_pixels, axis=0)

        # making prediction with model
        predictions = model.predict(img_pixels)[0]
        emotions=('angry', 'disgusted', 'scared', 'happy', 'sad', 'surprised', 'neutral')
        label=emotions[predictions.argmax()]
        
        cv2.putText(test_img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
    resized_img = cv2.resize(test_img, (1200, 800))
    cv2.imshow('Facial Emotion Recognition', resized_img)
    
    # 'q to exit'
    if cv2.waitKey(10) == ord('q'):  
        break

vid.release()
cv2.destroyAllWindows