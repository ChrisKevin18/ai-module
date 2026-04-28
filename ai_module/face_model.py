import os
import pickle
import numpy as np
import cv2

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

PROTO_PATH = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
MODEL_DNN_PATH = os.path.join(os.path.dirname(__file__), "res10.caffemodel")


class FaceRecognizer:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Face model not found.\nRun ai_module.train_faces() first."
            )

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        # Load DNN face detector
        self.net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_DNN_PATH)

    def recognize(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                try:
                    face = cv2.resize(face, (100, 100))
                except:
                    continue

                face_flat = face.flatten().reshape(1, -1)

                probs = self.model.predict_proba(face_flat)[0]
                idx = np.argmax(probs)

                name = self.model.classes_[idx]
                score = probs[idx]

                if score < self.threshold:
                    name = "Unknown"

                results.append({
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "name": name,
                    "score": score
                })

        return results