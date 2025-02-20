from flask import Flask, jsonify, request
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

@app.route('/crop_image', methods=['POST'])
def crop_image():
    # Load the image file from the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to load the image"}), 400

    # Convert to grayscale and apply a threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    thresh_inv = cv2.bitwise_not(thresh)

    # Detect horizontal lines
    kernel = np.ones((1, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)

    # Find contours after removing lines
    contours, _ = cv2.findContours(thresh_inv - horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop the image and encode in base64 if contours are found
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y + h, x:x + w]

        # Encode the cropped image to base64
        _, buffer = cv2.imencode('.png', cropped_image)
        cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "cropped_image": f"data:image/png;base64,{cropped_image_base64}"
        })
    else:
        return jsonify({"error": "No contours found to crop the image"}), 400

@app.route('/bone_length', methods=['POST'])
def bone_length():
    data = request.get_json()
    if 'cropped_image' not in data or 'pixel_diameter_ball' not in data:
        return jsonify({"error": "Cropped image and pixel diameter must be provided"}), 400

    # Decode the base64 image
    cropped_image_data = data['cropped_image'].split(",")[1]
    cropped_image_bytes = base64.b64decode(cropped_image_data)
    cropped_image_np = np.frombuffer(cropped_image_bytes, np.uint8)
    image = cv2.imdecode(cropped_image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to decode cropped image"}), 400

    # Convert to grayscale and find contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=20, threshold2=100)

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        output_image = image.copy()
        cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)

        hull = cv2.convexHull(largest_contour)

        # Find the two farthest points on the convex hull
        max_distance = 0
        pt1 = pt2 = None

        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                distance = np.linalg.norm(hull[i][0] - hull[j][0])
                if distance > max_distance:
                    max_distance = distance
                    pt1, pt2 = hull[i][0], hull[j][0]

        # Adjust max_distance as required
        max_distance += 25
        pixel_diameter_ball = float(data['pixel_diameter_ball'])
        ball_cm = 2.5  # Diameter of the ball in cm
        conversion_factor = max_distance / pixel_diameter_ball
        length_in_cm = conversion_factor * ball_cm

        # Draw a red line from pt1 to pt2
        if pt1 is not None and pt2 is not None:
            cv2.line(output_image, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

        # Encode the output image to base64
        _, buffer = cv2.imencode('.png', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the processed image and length as JSON
        return jsonify({
            "bone_image": f"data:image/png;base64,{output_image_base64}",
            "length_in_cm": length_in_cm
        })
    else:
        return jsonify({"error": "No contours found in the image."}), 404

if __name__ == '__main__':
    app.run(port=5001, debug=True)
