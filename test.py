import cv2
import pickle
import numpy as np
from skimage.transform import resize

model_path = 'C:/Users/chatt/Desktop/Projet Pfa/classification svm/model/Last Version/DecisionTreeClassifierCV.p'
scaler_path = 'C:/Users/chatt/Desktop/Projet Pfa/classification svm/model/scaler.p' 

with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

def preprocess_frame(frame):
    resized_frame = resize(frame, (100, 100), anti_aliasing=True, mode='reflect')
    flattened_frame = resized_frame.flatten().reshape(1, -1)
    normalized_frame = scaler.transform(flattened_frame)
    return normalized_frame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_texts = ["Nothing", "smoke", "fire"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)

    prediction = loaded_model.predict(processed_frame)[0]
    prediction_text = prediction_texts[prediction]

    cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
