import cv2
import pickle
import numpy as np
from skimage.transform import resize

# Définir les chemins vers le modèle, le scaler et l'image
model_path = 'C:/Users/chatt/Desktop/Projet Pfa/classification svm/model/Last Version/svmrbfCV.p'
scaler_path = 'C:/Users/chatt/Desktop/Projet Pfa/classification svm/model/scaler.p'  # Chemin vers le scaler
image_path = './Forest.jpg'

# Charger le modèle et le scaler
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Définir une fonction pour prétraiter l'image
def preprocess_frame(frame):
    resized_frame = resize(frame, (100, 100), anti_aliasing=True)  # Redimensionner l'image
    flattened_frame = resized_frame.flatten().reshape(1, -1)
    normalized_frame = scaler.transform(flattened_frame)  # Normaliser l'image
    return normalized_frame

# Charger et prétraiter l'image
image = cv2.imread(image_path)
if image is not None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
    processed_frame = preprocess_frame(image)

    # Faire la prédiction
    prediction = loaded_model.predict(processed_frame)[0]
    prediction_texts = ["Nothing", "smoke", "fire"]
    prediction_text = prediction_texts[prediction]

    # Afficher l'image et la prédiction
    cv2.putText(image, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convertir de RGB à BGR pour l'affichage

    # Attendre une touche pour fermer la fenêtre
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Erreur: L'image n'a pas pu être chargée.")
