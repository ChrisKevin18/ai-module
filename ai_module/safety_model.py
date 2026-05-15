import os
from ultralytics import YOLO
import cv2
import numpy as np

class SafetyDetector:

    def __init__(self, conf=0.4, imgsz=640):

        base_path = os.path.dirname(__file__)

        main_model_path = os.path.join(base_path, "best_old.pt")
        glove_model_path = os.path.join(base_path, "best.pt")

        if not os.path.exists(main_model_path):
            raise FileNotFoundError("best_old.pt not found")

        if not os.path.exists(glove_model_path):
            raise FileNotFoundError("best.pt not found")

        # Main PPE model
        self.main_model = YOLO(main_model_path)

        # Gloves + goggles model
        self.glove_model = YOLO(glove_model_path)

        self.conf = conf
        self.imgsz = imgsz

        print("MAIN MODEL:", self.main_model.names)
        print("GLOVE MODEL:", self.glove_model.names)

        self.classes = {
            "helmet": ["helmet"],
            "vest": ["vest"],
            "gloves": ["gloves"],
            "boots": ["boots"],
            "goggles": ["goggles"]
        }

    def detect(self, frame):

        frame = cv2.resize(frame, (640, 640))

        detected = set()

        # ---------------- MAIN PPE MODEL ----------------
        results_main = self.main_model(
            frame,
            conf=0.4,
            imgsz=640,
            verbose=False
        )[0]

        # ---------------- GLOVE MODEL ----------------
        results_glove = self.glove_model(
            frame,
            conf=0.75,
            imgsz=640,
            verbose=False
        )[0]

        # ---------------- PROCESS MAIN MODEL ----------------
        for box in results_main.boxes:

            cls_id = int(box.cls[0])
            label = self.main_model.names[cls_id].lower()
            conf = float(box.conf[0])

            print("MAIN:", label, conf)

            detected.add(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        # ---------------- PROCESS GLOVE MODEL ----------------
        # ---------------- PROCESS GLOVE MODEL ----------------
        THRESHOLDS = {
            "gloves": 0.75,
            "goggles": 0.50
        }

        for box in results_glove.boxes:

            cls_id = int(box.cls[0])
            label = self.glove_model.names[cls_id].lower()
            conf = float(box.conf[0])

            print("SPECIAL:", label, conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            # ---------------- STRICT FILTER FOR GOGGLES ----------------
            if label == "goggles":

                # Reject weak confidence
                if conf < THRESHOLDS["goggles"]:
                    continue

                # Reject tiny/random eye detections
                if w < 40 or h < 15:
                    continue

            # ---------------- STRICT FILTER FOR GLOVES ----------------
            if label == "gloves":

                if conf < THRESHOLDS["gloves"]:
                    continue

            # VALID DETECTION
            detected.add(label)

            color = (255,0,255) if label == "gloves" else (0,255,255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("FRAME", frame)
        cv2.waitKey(1)

        return detected

    def check_all(self, frame):

        detected = self.detect(frame)

        return {
            "helmet": "helmet" in detected,
            "vest": "vest" in detected,
            "gloves": "gloves" in detected,
            "boots": "boots" in detected,
            "goggles": "goggles" in detected
        }