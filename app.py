import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Fish or Dog Image Recognition")
st.write("Upload an image and the deep learning model will classify it as a dog or a fish.")

# Load model
model = tf.keras.models.load_model("fish_dog_cnn_model.h5")

# Upload image
uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read and display image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    # Preprocess
    img_resized = img.resize((128,128))
    img_arr = np.array(img_resized).astype("float32") / 255.0
    img_arr = img_arr.reshape(1,128,128,3)

    # Predict
    prediction = model.predict(img_arr)[0][0]

    if prediction > 0.5:
        label = "DOG"
    else:
        label = "FISH"

    st.subheader("Prediction:")
    st.success(label)

    st.write(f"Model confidence: {prediction:.4f}")
