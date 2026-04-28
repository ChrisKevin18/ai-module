import os
import logging

from .logger import setup_logger
from .face_utils import capture_faces_internal, train_faces_internal
from .server import main


# -------- PATH -------- #
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


def model_exists():
    return os.path.exists(MODEL_PATH)


# -------- MAIN RUN -------- #

def run(camera=None, threshold=0.6, ppe=True, auto_setup=True):
    """
    Main entry point

    Args:
        camera: None (auto detect) OR int (webcam) OR str (RTSP/IP)
        threshold: face recognition threshold
        ppe: enable PPE detection
        auto_setup: auto capture + train if model not found
    """

    setup_logger()

    # -------- FIRST TIME SETUP -------- #
    if auto_setup and not model_exists():
        logging.warning("⚠ No trained model found. First-time setup starting...")

        names = []

        while True:
            name = input("Enter person name (or 'done'): ").strip()

            if name.lower() == "done":
                break

            if name == "":
                print("⚠ Name cannot be empty")
                continue

            logging.info(f"📸 Capturing data for {name}...")
            capture_faces_internal(name)
            names.append(name)

        # -------- VALIDATION -------- #
        if len(names) < 2:
            logging.error("❌ Need at least 2 persons to train model")
            logging.info("👉 Restart and add more users")
            return

        # -------- TRAIN -------- #
        logging.info("📚 Training model...")
        train_faces_internal()
        logging.info("✅ Setup complete!")

    # -------- RUN SYSTEM -------- #
    logging.info("🚀 Starting AI System...")

    main(
        camera=camera,        # None → auto detect
        threshold=threshold,
        ppe=ppe
    )


# -------- OPTIONAL HELPERS -------- #

def run_face(camera=None, threshold=0.6):
    """
    Run only face recognition
    """
    run(camera=camera, threshold=threshold, ppe=False)


def run_ppe(camera=None):
    """
    Run only PPE detection
    """
    run(camera=camera, threshold=0, ppe=True)