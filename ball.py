from flask import Flask, jsonify
import cv2
import numpy as np
import os
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure folder for processed images
PROCESSED_FOLDER = os.path.join('static', 'processed')
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure processed folder exists
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/ball_image', methods=['POST'])
def ball_image():
    # Load the corrected image from the processed folder
    image_path = os.path.join(app.config['PROCESSED_FOLDER'], "corrected.png")
    if not os.path.exists(image_path):
        return jsonify({"error": "Corrected image not found"}), 404

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Failed to load corrected image"}), 400

    # Detect circles in the image
    detected_circles = preprocess_image_for_circle_detection(image)

    # Initialize variable to store the largest ball diameter in pixels
    largest_ball_diameter = 0

    # If circles are found, draw them and calculate diameters
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            center = (circle[0], circle[1])  # Circle center
            radius = circle[2]               # Circle radius
            ball_diameter = radius * 2       # Calculate diameter in pixels

            # Update largest ball diameter if the current one is larger
            if ball_diameter > largest_ball_diameter:
                largest_ball_diameter = ball_diameter

            # Draw the circle and center point on the image
            cv2.circle(image, center, radius, (255), 3)  # White circle
            cv2.circle(image, center, 2, (255), 3)       # White center point

            # Annotate the diameter on the image
            cv2.putText(image, f'D: {ball_diameter} px', (center[0] - 20, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 2)  # White text for diameter

    # Encode the processed image to base64
    _, buffer = cv2.imencode('.png', image)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"ball_image": f"data:image/png;base64,{processed_image_base64}"})

def preprocess_image_for_circle_detection(image):
    # Convert to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    return circles

if __name__ == '__main__':
    app.run(port=5001, debug=True)
