import cv2
import threading
import os

# -------- FORCE TCP FOR RTSP -------- #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# -------- AUTO DETECT CAMERA -------- #

def get_camera_source():
    print("[INFO] Detecting camera...")

    # 1️⃣ Try local cameras first
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[INFO] Using Local Camera {i}")
            return i

    # 2️⃣ Try RTSP (best for CCTV / long distance)
    rtsp = os.environ.get("RTSP_URL")
    if rtsp:
        print("[INFO] Using RTSP Camera (TCP)")
        return rtsp

    # 3️⃣ Try HTTP IP Camera (mobile apps, etc.)
    ip = os.environ.get("IP_CAMERA_URL")
    if ip:
        print("[INFO] Using IP Camera (HTTP)")
        return ip

    raise RuntimeError("❌ No camera found")



def get_all_sources():
    sources = []

    # 1️⃣ Local cameras
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[INFO] Found Local Camera {i}")
            sources.append(i)

    # 2️⃣ RTSP cameras (comma separated)
    rtsp_list = os.environ.get("RTSP_URLS")
    if rtsp_list:
        for url in rtsp_list.split(","):
            print(f"[INFO] Found RTSP: {url}")
            sources.append(url.strip())

    # 3️⃣ IP cameras
    ip_list = os.environ.get("IP_CAMERA_URLS")
    if ip_list:
        for url in ip_list.split(","):
            print(f"[INFO] Found IP Camera: {url}")
            sources.append(url.strip())

    if not sources:
        raise RuntimeError("❌ No cameras found")

    return sources
# -------- THREADED CAMERA -------- #

class CameraStream:
    def __init__(self, src=None):

        if src is None:
            src = get_camera_source()

        # Backend selection
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Failed to open camera")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.cap.grab()
            self.ret, self.frame = self.cap.retrieve()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()



    