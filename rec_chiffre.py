import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

# Charger le modèle entraîné
model = keras.models.load_model('./mnist_cnn.h5')

# Définir la fonction de prédiction
def predict(image):
    # Prétraitement de l'image
    image = cv2.resize(image, (28, 28)) # redimensionner l'image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convertir l'image en niveaux de gris
    image = image.reshape(1, 28, 28, 1) # ajouter une dimension pour le canal
    image = image.astype('float32') / 255.0 # normaliser l'image, float32 permet d'éviter une erreur de type lors de la normalisation
    # astype('float32') permet de convertir l'image en float32


    # Prédiction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return predicted_class

# Créer l'application Streamlit
def main():
    # centrer l'application
    st.set_page_config(layout="centered")
    # centrer le titre
    st.markdown("<h1 style='text-align: center;'>Reconnaissance de chiffres manuscrits</h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Charger une image", type=["png", "jpg"])

    if uploaded_image is not None:
        # Lire l'image téléchargée
        #image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        st.image(image, caption='Image chargée', use_column_width=True)

        # Prédire le chiffre
        predicted_digit = predict(image)
        st.write("Chiffre prédit :", predicted_digit)

if __name__ == '__main__':
    main()
