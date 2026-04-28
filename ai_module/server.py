import cv2
import logging
import threading
from multiprocessing import Process
from queue import Queue

from ai_module.face_model import FaceRecognizer
from ai_module.safety_model import SafetyDetector
from ai_module.camera import CameraStream, get_all_sources
from ai_module.config import *


# -------- WORKER THREAD -------- #

def inference_worker(frame_queue, result_queue, face, safety, ppe):
    while True:
        frame = frame_queue.get()

        if frame is None:
            break

        faces = face.recognize(frame) if face else []
        safety_res = safety.check_all(frame) if ppe and safety else {}

        result_queue.put((faces, safety_res))


# -------- CAMERA PROCESS -------- #

def process_camera(source, cam_id, threshold, ppe):

    logging.info(f"[CAM {cam_id}] Starting...")

    cam = CameraStream(source)

    face = FaceRecognizer(threshold) if threshold > 0 else None
    safety = SafetyDetector() if ppe else None

    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)

    worker = threading.Thread(
        target=inference_worker,
        args=(frame_queue, result_queue, face, safety, ppe),
        daemon=True
    )
    worker.start()

    cached_faces = []
    cached_safety = {}

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Send frame (non-blocking)
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get results
        if not result_queue.empty():
            cached_faces, cached_safety = result_queue.get()

        # -------- DRAW FACE -------- #
        for res in cached_faces:
            x, y, w, h = res["box"]
            name = res["name"]
            score = res["score"]

            conf = score * 10
            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{name} ({conf:.1f}/10)",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # -------- DRAW PPE -------- #
        y_offset = 20
        missing = []

        for k, v in cached_safety.items():
            color = (0,255,0) if v else (0,0,255)

            if not v:
                missing.append(k.upper())

            cv2.putText(frame, f"{k.upper()}: {'YES' if v else 'NO'}",
                        (10,y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            y_offset += 20

        cv2.imshow(f"Camera {cam_id}", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


# -------- MAIN MULTI-CAMERA -------- #

def main(camera=None, threshold=0.6, ppe=True):

    import logging
    logging.info("Starting Multi-Camera AI System...")


    if camera is not None:
        sources = [camera]
    else:
        sources = get_all_sources()

    processes = []

    for i, src in enumerate(sources):
        p = Process(
            target=process_camera,
            args=(src, i, threshold, ppe)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()