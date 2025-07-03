import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# Path to Haar cascade (included with OpenCV)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Your dataset directory
DATA_DIR = "C:\Users\Administrator\Downloads\python\image_classification_dataset"

IMAGE_SIZE = (100, 100)  # Resize face

def detect_face(img_path):  
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, IMAGE_SIZE)
    return face_resized.flatten()  # Flatten to 1D vector

# Load images and labels
def load_data():
    features, labels = [], []
    class_names = os.listdir(DATA_DIR)

    for label, person in enumerate(class_names):
        person_dir = os.path.join(DATA_DIR, person)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            face_vector = detect_face(img_path)
            if face_vector is not None:
                features.append(face_vector)
                labels.append(label)

    return np.array(features), np.array(labels), class_names

# Load data
print("Loading data...")
X, y, class_names = load_data()
print(f"Classes: {class_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Predict test image
test_img = "test.jpg"  # ← Change to your own test image path
test_face = detect_face(test_img)
if test_face is not None:
    prediction = model.predict([test_face])
    print("Predicted class:", class_names[prediction[0]])
    img = Image.open(test_img).resize((200, 200))
    img.show()
else:
    print("No face found in test image.")
