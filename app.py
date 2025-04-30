import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

# Load the trained model
model = tf.keras.models.load_model("retina_classifier.h5")

# Define class labels
class_labels = ['0', '1']  # Modify based on your dataset

# Function to preprocess an image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    return image

# Streamlit UI
st.title("Retinal Disease Classification")
st.write("Upload an image to classify retinal disease")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Display result
    if predicted_class == 0:
        st.success("Predicted Retinal Disease Test: 0")
    else:
        st.error("Predicted Retinal Disease Test: 1")
