import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Modell laden
model = load_model("model/keras_model.h5", compile=False)

# Labels laden
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(image: Image.Image):
    # Bild vorbereiten
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = image / 255.0  # Normalisierung

    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    label = class_names[index]

    return label, confidence
