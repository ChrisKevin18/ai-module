import cv2
import threading
import os

# -------- FORCE TCP FOR RTSP -------- #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# -------- AUTO DETECT CAMERA -------- #
def get_camera_source():

    print("[INFO] Detecting selected camera source...")

    # -----------------------------------
    # RTSP CAMERA
    # -----------------------------------

    rtsp_urls = os.environ.get("RTSP_URLS")

    if rtsp_urls:

        first_rtsp = rtsp_urls.split(",")[0].strip()

        print(f"[INFO] Selected RTSP Stream: {first_rtsp}")

        test_cap = cv2.VideoCapture(first_rtsp)

        if not test_cap.isOpened():

            raise RuntimeError(
                f"❌ Failed to open RTSP stream:\n{first_rtsp}"
            )

        test_cap.release()

        return first_rtsp

    # -----------------------------------
    # HTTP/IP CAMERA
    # -----------------------------------

    ip_urls = os.environ.get("IP_CAMERA_URLS")

    if ip_urls:

        first_ip = ip_urls.split(",")[0].strip()

        print(f"[INFO] Selected IP Camera: {first_ip}")

        test_cap = cv2.VideoCapture(first_ip)

        if not test_cap.isOpened():

            raise RuntimeError(
                f"❌ Failed to open IP Camera:\n{first_ip}"
            )

        test_cap.release()

        return first_ip

    # -----------------------------------
    # LOCAL WEBCAM
    # -----------------------------------

    webcam_index = 0

    print(f"[INFO] Selected Webcam: {webcam_index}")

    test_cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)

    if not test_cap.isOpened():

        raise RuntimeError(
            "❌ Failed to open local webcam"
        )

    test_cap.release()

    return webcam_index



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



    