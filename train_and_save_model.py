import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# Configuration
DATA_DIR = "C:\\Users\\keno\\Desktop\\python\\image_classification_dataset"

IMAGE_SIZE = (100, 100)
MODEL_PATH = "knn_model.joblib"
LABELS_PATH = "class_names.txt"

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, IMAGE_SIZE).flatten()

# Load and label faces
features, labels = [], []
class_names = os.listdir(DATA_DIR)

for idx, class_name in enumerate(class_names):
    person_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        vec = detect_face(img_path)
        if vec is not None:
            features.append(vec)
            labels.append(idx)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, labels)

# Save model and labels
dump(model, MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    f.write("\n".join(class_names))

print("✅ Model and class names saved!")
