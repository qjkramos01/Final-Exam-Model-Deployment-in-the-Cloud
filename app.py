import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("MNIST Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit.")

model = tf.keras.models.load_model("mnist_cnn_model.h5")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    st.image(image, caption='Uploaded Image', width=200)
    
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(img_array))

    st.write("### Predicted Digit:", prediction)
