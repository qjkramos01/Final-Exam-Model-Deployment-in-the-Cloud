import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Dog vs Fish Image Recognition")
st.write("Upload an image and the model will classify it as a dog or a fish.")

# Load model
model = tf.keras.models.load_model("fish_dog_cnn_model.h5")

# Upload image
uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    # Preprocess
    img_resized = img.resize((64,64))
    img_arr = np.array(img_resized).astype("float32") / 255.0
    img_arr = img_arr.reshape(1,64,64,3)

    # Predict
    pred = model.predict(img_arr)[0][0]
    label = "DOG" if pred>0.5 else "FISH"

    st.subheader("Prediction:")
    st.success(label)
    st.write(f"Confidence: {pred:.4f}")
