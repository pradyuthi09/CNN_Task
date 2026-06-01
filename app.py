import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("plant_disease_model.h5")

# =========================
# CLASS NAMES
# Replace with your classes
# =========================
class_names = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot"
]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌿 Plant Disease Detection")

st.sidebar.info("""
### About Project

This CNN-based application predicts plant leaf diseases from uploaded images.

#### Features
✔ Disease Detection

✔ Confidence Score

✔ Image Classification

✔ Fast Prediction

✔ Farmer Assistance
""")

# =========================
# MAIN HEADER
# =========================
st.title("🌱 AI-Powered Plant Disease Detection System")

st.markdown("""
Upload a plant leaf image and the CNN model will analyze it to identify the disease.
""")

st.markdown("---")

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION SECTION
# =========================
if uploaded_file is not None:

    try:

        # Convert image to RGB
        image = Image.open(uploaded_file).convert("RGB")

        # Two Column Layout
        col1, col2 = st.columns([1, 1])

        # LEFT COLUMN
        with col1:

            st.subheader("📷 Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        # PREPROCESS IMAGE
        img = image.resize((128, 128))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # PREDICTION
        prediction = model.predict(img)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        # RIGHT COLUMN
        with col2:

            st.subheader("🩺 Disease Analysis Report")

            st.metric(
                label="Detected Disease",
                value=predicted_class
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.progress(int(confidence))

            if confidence > 80:
                st.success("High Confidence Prediction")
            elif confidence > 60:
                st.warning("Moderate Confidence Prediction")
            else:
                st.error("Low Confidence Prediction")

        st.markdown("---")

        # PROBABILITY TABLE
        st.subheader("📊 Class Probability Distribution")

        for i, class_name in enumerate(class_names):

            prob = float(prediction[0][i]) * 100

            st.write(
                f"**{class_name}** : {prob:.2f}%"
            )

            st.progress(min(int(prob), 100))

    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:

    st.info("Please upload a plant leaf image to get started.")
    