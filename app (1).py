import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# 1. Function to load image as grayscale
# ----------------------------
def load_image_to_gray(file):
    img = Image.open(file)
    img = img.convert("L")  # grayscale
    return np.array(img)

# ----------------------------
# 2. Streamlit App
# ----------------------------
st.title("Fish or Dog Image Recognition")
st.write("Upload an image and the deep learning model will classify it as a dog or a fish.")

# Load model
model = tf.keras.models.load_model("fish_dog_cnn_model.h5")

# Upload image
uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img_array = load_image_to_gray(uploaded)

    # Resize to 40x40 (model input)
    img_resized = Image.fromarray(img_array).resize((40,40))
    img_arr = np.array(img_resized).astype("float32") / 255.0

    # Reshape for model: (1, 40, 40, 1)
    img_arr = img_arr.reshape(1,40,40,1)

    # Display
    st.image(img_resized, width=150, caption="Uploaded Image", clamp=True)

    # Predict
    prediction = model.predict(img_arr)[0][0]

    if prediction > 0.5:
        label = "DOG"
    else:
        label = "FISH"

    st.subheader("Prediction:")
    st.success(label)
    st.write(f"Model confidence: {prediction:.4f}")
