from flask import Flask, jsonify, request
from motion_detector import MotionDetector
import threading
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Create a single shared detector instance
detector = MotionDetector()

@app.route('/motion-status', methods=['GET'])
def motion_status():
    status = detector.get_status()
    print(f"[API] Motion status sent: {status}")
    return jsonify({"motion_status": status})


if __name__ == '__main__':
    # Start motion detection thread before Flask app
    thread = threading.Thread(target=detector.start_detection)
    thread.daemon = True
    thread.start()

    app.run(debug=True, use_reloader=False)
