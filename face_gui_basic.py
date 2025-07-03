import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from joblib import load
import os

# === STREAMLIT CONFIG ===
st.set_page_config(
    page_title="Face Classifier  App",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === CONSTANTS ===
MODEL_PATH = "knn_model.joblib"
LABELS_PATH = "class_names.txt"
IMAGE_SIZE = (100, 100)

# === LOAD MODEL AND LABELS ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

try:
    model = load(MODEL_PATH)
except FileNotFoundError:
    st.error("🚫 Model file 'knn_model.joblib' not found!")
    st.stop()

try:
    with open(LABELS_PATH, "r") as f:
        class_names = f.read().splitlines()
except FileNotFoundError:
    st.error("🚫 Class labels file 'class_names.txt' not found!")
    st.stop()

# === FACE DETECTION FUNCTION ===
def detect_face_from_uploaded(image):
    try:
        img = np.array(image.convert('RGB'))
    except UnidentifiedImageError:
        return None, None, "Invalid image format."
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, img, "No face detected."

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, IMAGE_SIZE).flatten()

    return face_resized, img[y:y+h, x:x+w], None

# === SIDEBAR UI ===
with st.sidebar:
    st.image("https://img.icons8.com/color/96/face-id.png", width=80)
    st.title("Face Recognition App")
    st.markdown("🔍 Upload a face image to recognize the person using a trained KNN model.")
    st.markdown("📁 **Model**: `knn_model.joblib`")
    st.markdown("📄 **Labels**: `class_names.txt`")
    st.markdown("---")
    st.caption("Built with OpenCV, PIL, Joblib & ❤️")

# === MAIN UI ===
st.markdown("<h1 style='text-align: center;'>🧠  Face classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a face photo to identify who it is.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='📷 Uploaded Image', use_column_width=True)

        with st.spinner("🔍 Detecting and Recognizing..."):
            face_vector, face_crop, error_msg = detect_face_from_uploaded(image)

            if error_msg:
                st.error(f"❌ {error_msg}")
            else:
                prediction = model.predict([face_vector])[0]
                try:
                    predicted_name = class_names[int(prediction)]
                except (IndexError, ValueError):
                    predicted_name = "Unknown"

                st.success(f"✅ Predicted Person: **{predicted_name}**")
                st.image(face_crop, caption="🧠 Detected Face", width=180)

    except UnidentifiedImageError:
        st.error("❌ Could not open image. Please upload a valid image file.")

else:
    st.info("📤 Please upload a `.jpg`, `.jpeg`, or `.png` image to begin.")

# === FOOTER ===
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 14px;'>Built with ❤️ </p>
    """,
    unsafe_allow_html=True
)
