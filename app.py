import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_image_to_rgb(file):
    # Open image
    img = Image.open(file)
    
    # Force convert to RGB (3 channels)
    img = img.convert("RGB")
    
    # Convert to NumPy array
    img_array = np.array(img)
    
    return img_array

st.title("Fish or Dog Image Recognition")
st.write("Upload an image and the deep learning model will classify it as a dog or a fish.")

model = tf.keras.models.load_model("fish_dog_cnn_model.h5")

uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img_array = load_image_to_rgb(uploaded)

    img_resized = Image.fromarray(img_array).resize((128,128))
    img_arr = np.array(img_resized).astype("float32") / 255.0
    img_arr = img_arr.reshape(1,128,128,3)

    st.image(img_resized, width=300, caption="Uploaded Image")

    prediction = model.predict(img_arr)[0][0]
    
    if prediction > 0.5:
        label = "DOG"
    else:
        label = "FISH"

    st.subheader("Prediction:")
    st.success(label)
    st.write(f"Model confidence: {prediction:.4f}")

