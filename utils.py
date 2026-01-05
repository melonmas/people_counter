import cv2
import numpy as np

def draw_roi(frame, roi_points):
    """
    Draws the Region of Interest (ROI) polygon on the frame.
    roi_points: List of (x, y) tuples.
    """
    if len(roi_points) > 2:
        cv2.polylines(frame, [np.array(roi_points)], True, (0, 255, 255), 2)

def draw_text_with_background(frame, text, x, y, font_scale=0.6, thickness=1, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draws text with a background rectangle for better visibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
