import cv2
import threading
import time

class VideoCamera(object):
    def __init__(self, src=0):
        # Open video source (0 for webcam, or RTSP url)
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.video.isOpened():
            raise ValueError("Could not open video source")

        # Read first frame
        (self.grabbed, self.frame) = self.video.read()

        # Threading setup
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start background frame reading
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.stop()
        if self.video.isOpened():
            self.video.release()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.video.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                # If stream ends/disconnects, stop
                self.stop()

    def get_frame(self):
        with self.lock:
            # Return a copy of the frame to avoid race conditions
            if self.frame is None:
                return None
            return self.frame.copy()
