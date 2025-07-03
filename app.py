import streamlit as st
import cv2
import numpy as np
from PIL import Image
from joblib import load
import os

# Load model and class names
MODEL_PATH = "knn_model.joblib"
LABELS_PATH = "class_names.txt"
IMAGE_SIZE = (100, 100)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = load(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = f.read().splitlines()

def detect_face_from_uploaded(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, img
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, IMAGE_SIZE).flatten()
    return face_resized, img[y:y+h, x:x+w]

def main():
    st.title("🧠 Face Recognition Web App")
    st.write("Upload an image with a clear face to predict the person.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Analyzing..."):
            face_vector, face_crop = detect_face_from_uploaded(image)

            if face_vector is None:
                st.error("No face detected.")
            else:
                prediction = model.predict([face_vector])
                predicted_name = class_names[prediction[0]]

                st.success(f"Predicted Person: **{predicted_name}**")
                st.image(face_crop, caption="Detected Face", width=150)

if __name__ == "__main__":
    main()
