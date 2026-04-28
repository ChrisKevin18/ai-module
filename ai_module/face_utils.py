import os
import cv2
import pickle
import logging
import numpy as np
from sklearn.svm import SVC

BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


def capture_faces_internal(name):
    path = os.path.join(DATASET_PATH, name)
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    logging.info("Press S to capture, Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(f"{path}/{count}.jpg", frame)
            logging.info(f"Saved {count}.jpg")
            count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_faces_internal():
    X = []
    y = []
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (100, 100)).flatten()

            X.append(image)
            y.append(person)

    X = np.array(X)

    model = SVC(probability=True)
    
    if len(set(y)) < 2:
     raise ValueError("Need at least 2 persons to train model")
    

    model.fit(X, y)
    
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logging.info("Model trained and saved")