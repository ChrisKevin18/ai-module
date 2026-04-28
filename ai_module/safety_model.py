import os
from ultralytics import YOLO
import cv2

class SafetyDetector:
    def __init__(self, conf=0.4, imgsz=320):

        model_path = os.path.join(os.path.dirname(__file__), "best.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError("best.pt not found inside ai_module")

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

        self.classes = {
            "helmet": ["helmet"],
            "vest": ["vest"],
            "gloves": ["gloves"],
            "boots": ["boots"],
            "goggles": ["goggles"]
        }

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        detected = set()

        for box in results.boxes:
            if box.conf[0] < self.conf:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id].lower()
            detected.add(label)

            # 🔥 Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0,255,0), 2)

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