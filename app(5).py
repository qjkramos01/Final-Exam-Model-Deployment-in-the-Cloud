import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
# Sets the page title, icon, and layout
st.set_page_config(
    page_title="Dog vs Fish Classifier",
    page_icon="🐾",
    layout="wide"
)

# --- Custom CSS Styling ---
# This function injects custom CSS into the Streamlit app
def load_css():
    st.markdown("""
    <style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Main content block - the "card" */
    [data-testid="stBlockContainer"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 2.5rem;
        margin-top: 2rem; /* Give space from top */
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    /* Style the title */
    h1 {
        color: #1a1a2e; /* Darker text */
        text-align: center;
        font-family: 'Arial', sans-serif;
    }

    /* Style the file uploader button */
    [data-testid="stFileUploader"] button {
        background-color: #0072ff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #00c6ff;
        color: white;
        border: 1px solid #00c6ff;
    }

    /* Style the st.progress bar */
    [data-testid="stProgressBar"] > div {
        background-color: #0072ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the custom CSS
load_css()

# --- App Title ---
st.title("🐾 Dog vs Fish 🐟 Image Recognition")
st.markdown("<h4 style='text-align: center; color: #555;'>Upload an image and the model will classify it!</h4>", unsafe_allow_html=True)
st.markdown("---") # Adds a horizontal line

# --- Model Loading ---
# Load the pre-trained model with error handling
try:
    with st.spinner("Loading classification model..."):
        model = tf.keras.models.load_model("fish_dog_cnn_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Have you run the 'train.py' script first to create the .h5 file?")
    st.stop()


# --- File Uploader ---
uploaded = st.file_uploader("Upload your image here...", type=["jpg", "jpeg", "png"])

if uploaded:
    # --- Image Preprocessing ---
    # Open the image, convert to RGB (for consistency)
    img = Image.open(uploaded).convert("RGB")
    
    # Resize to the model's expected input size (64x64)
    img_resized = img.resize((64, 64))
    
    # Convert image to numpy array, normalize pixel values
    img_arr = np.array(img_resized).astype("float32") / 255.0
    
    # Reshape to match the model's input shape (1, 64, 64, 3)
    img_arr = img_arr.reshape(1, 64, 64, 3)

    # --- Prediction ---
    with st.spinner("Analyzing image..."):
        pred = model.predict(img_arr)[0][0]

    # --- Interpret Prediction ---
    if pred > 0.5:
        label = "FISH 🐟"
        confidence = pred * 100
        color = "#e76f51" # Coral for fish
        bg_color = "#fef6f4"
    else:
        label = "DOG 🐶"
        confidence = (1 - pred) * 100
        color = "#2a9d8f" # Teal for dog
        bg_color = "#eafaf8"

    # --- Display Results ---
    # Use columns for a side-by-side layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Your Upload", use_column_width=True, channels="RGB")

    with col2:
        st.subheader("🔮 Prediction")
        
        # Custom-styled prediction box
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border: 2px solid {color};
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            margin: 10px 0;
        ">
            <h2 style="color: {color}; margin: 0; font-size: 2.5rem;">{label}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Confidence")
        st.progress(int(confidence))
        st.markdown(f"<h3 style='text-align: center; color: #333;'>{confidence:.2f}%</h3>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to get a prediction.")
