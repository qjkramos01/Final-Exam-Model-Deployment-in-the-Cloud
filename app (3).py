# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Dog vs Fish Image Recognition")
st.write("Upload an image and the model will classify it as a dog or a fish.")

# Load model
# This file MUST exist. You create it by running train.py
try:
    model = tf.keras.models.load_model("fish_dog_cnn_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Have you run the 'train.py' script first to create the .h5 file?")
    st.stop()


# Upload image
uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    # Preprocess
    img_resized = img.resize((64,64))
    img_arr = np.array(img_resized).astype("float32") / 255.0
    img_arr = img_arr.reshape(1,64,64,3) # Add batch dimension

    # Predict
    pred = model.predict(img_arr)[0][0]
    
    # --- CORRECTED LOGIC ---
    # 'dog' = 0, 'fish' = 1
    # A prediction > 0.5 means the model is leaning towards 1 (FISH)
    if pred > 0.5:
        label = "FISH"
        confidence = pred
    else:
        label = "DOG"
        confidence = 1 - pred # Show confidence for the predicted class

    st.subheader("Prediction:")
    st.success(label)
    st.write(f"Confidence: {confidence:.4f}")
