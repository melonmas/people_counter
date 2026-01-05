from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from camera import VideoCamera
from tracker import CentroidTracker
from utils import draw_text_with_background

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# Initialize MediaPipe Object Detector
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5 
)
detector = vision.ObjectDetector.create_from_options(options)

# Global Variables
camera = None
tracker = CentroidTracker(maxDisappeared=30, maxDistance=80)
total_visitors = 0
current_inside = 0

# Trackable Objects Dictionary
trackableObjects = {}

# Proximity Zone Configuration
PROXIMITY_RATIO = 0.6 
MIN_AREA_RATIO = 0.15 # 15% of screen area
REQUIRED_DURATION = 10 # Seconds to wait before counting

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

def process_frame(frame):
    global total_visitors, current_inside, trackableObjects
    
    h, w, c = frame.shape
    screen_area = w * h
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    rects = []
    
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == 'person':
            bbox = detection.bounding_box
            
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w_box = int(bbox.width)
            h_box = int(bbox.height)
            
            x = max(0, x)
            y = max(0, y)
            endX = min(w, x + w_box)
            endY = min(h, y + h_box)
            
            rects.append((x, y, endX, endY))
            
            box_area = w_box * h_box
            ratio = box_area / screen_area
            
            color = (0, 0, 255) # Red (Far)
            if ratio > MIN_AREA_RATIO:
                color = (0, 255, 255) # Yellow (Pending)
                
            cv2.rectangle(frame, (x, y), (endX, endY), color, 2)
            cv2.putText(frame, f"{int(ratio*100)}% Area", (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ------------------ TRACKING LOGIC ------------------
    objects = tracker.update(rects)
    
    draw_text_with_background(frame, f"REQ SIZE > {int(MIN_AREA_RATIO*100)}% | TIME > {REQUIRED_DURATION}s", 10, h - 10, bg_color=(0, 0, 0))
    
    current_inside = 0 
    
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        
        if to is None:
            # Initialize with enter_time = None
            to = {
                'centroids': [centroid], 
                'counted': False, 
                'max_area_ratio': 0,
                'enter_time': None
            }
        else:
            to['centroids'].append(centroid)
        
        # Update Area Ratio based on closest box
        current_area_ratio = 0
        for (rx, ry, rex, rey) in rects:
            rcX = (rx + rex) // 2
            rcY = (ry + rey) // 2
            dist = np.sqrt((centroid[0]-rcX)**2 + (centroid[1]-rcY)**2)
            if dist < 50: 
                 area = (rex-rx) * (rey-ry)
                 current_area_ratio = area / screen_area
        
        # Update Prop
        if current_area_ratio > to['max_area_ratio']:
            to['max_area_ratio'] = current_area_ratio
            
        color = (0, 0, 255) # Default Red
        
        # ------------------ DWELL TIME COUNTING ------------------
        # Condition: Must be currently BIG (Close)
        if current_area_ratio > MIN_AREA_RATIO:
            # Start timer if not started
            if to['enter_time'] is None:
                to['enter_time'] = time.time()
            
            elapsed = time.time() - to['enter_time']
            
            if not to['counted']:
                color = (0, 255, 255) # Yellow = Counting down
                remaining = max(0, REQUIRED_DURATION - elapsed)
                
                # Visual Timer above head
                cv2.putText(frame, f"{remaining:.1f}s", (centroid[0], centroid[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if elapsed >= REQUIRED_DURATION:
                    total_visitors += 1
                    to['counted'] = True
            else:
                color = (0, 255, 0) # Green = COUNTED
                
            current_inside += 1
            
        else:
            # If they shrink/move away, RESET the timer?
            # Yes, "diam selama 10 detik" implies continuous presence.
            if not to['counted']:
                to['enter_time'] = None
            else:
                color = (0, 255, 0) # Already counted, keep green
        
        trackableObjects[objectID] = to
        
        text = f"ID {objectID}"
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    draw_text_with_background(frame, f"Total Pengunjung: {total_visitors}", 10, 40, font_scale=0.8, thickness=2, bg_color=(255, 0, 0))

    return frame

def generate_frames():
    cam = get_camera()
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        'total_in': total_visitors, 
        'total_out': 0,
        'current_inside': current_inside
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
