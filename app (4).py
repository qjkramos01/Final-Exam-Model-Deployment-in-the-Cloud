# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Dog vs Fish Image Recognition")
st.write("Upload an image and the model will classify it as a dog or a fish.")

# --- 1. Load the Trained Model ---
# This file MUST exist. You create it by running train.py
try:
    model = tf.keras.models.load_model("fish_dog_cnn_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Have you run the 'train.py' script first to create the .h5 file?")
    st.stop()


# --- 2. Image Upload ---
uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    # --- 3. Preprocess the Image ---
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    # Resize to the model's expected input size (64x64)
    img_resized = img.resize((64,64))
    
    # Convert to numpy array and rescale (0-1)
    img_arr = np.array(img_resized).astype("float32") / 255.0
    
    # Add a batch dimension (model expects 4D tensor)
    img_arr = img_arr.reshape(1, 64, 64, 3) 

    # --- 4. Make Prediction ---
    # The output is a number between 0 (dog) and 1 (fish)
    pred = model.predict(img_arr)[0][0]
    
    # --- 5. Display Corrected Logic ---
    # Based on 'flow_from_directory': 'dog' = 0, 'fish' = 1
    # A prediction > 0.5 means the model is leaning towards 1 (FISH)
    
    if pred > 0.5:
        label = "FISH"
        confidence = pred * 100 # Percentage confidence for FISH
    else:
        label = "DOG"
        confidence = (1 - pred) * 100 # Percentage confidence for DOG

    st.subheader("Prediction:")
    st.success(f"{label}")
    st.write(f"Confidence: {confidence:.2f}%")
