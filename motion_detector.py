import cv2
import threading
import time

class MotionDetector:
    def __init__(self):
        self.motion_status = "No Motion Detected"
        self.lock = threading.Lock()
        self.last_motion_time = time.time()
        self.running = True

    def get_status(self):
        with self.lock:
            return self.motion_status

    def start_detection(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Unable to access camera")
            return

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        while self.running:
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) > 1000 for c in contours)
            now = time.time()

            with self.lock:
                if motion_detected:
                    self.motion_status = "Motion Detected"
                    self.last_motion_time = now
                elif now - self.last_motion_time > 2:
                    self.motion_status = "No Motion Detected"

            # Display motion status on the preview window
            cv2.putText(frame1, self.motion_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the frame with overlay
            cv2.imshow("Motion Detection Preview", frame1)

            frame1 = frame2
            ret, frame2 = cap.read()
            if not ret:
                break

            # Exit preview on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            print(f"[Motion] {self.motion_status}")

        cap.release()
        cv2.destroyAllWindows()

