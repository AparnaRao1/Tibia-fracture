from flask import Flask, jsonify, request, url_for, send_from_directory
import cv2
import numpy as np
import os
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure folders for uploads and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'  # Ensure this is a static folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/crop_image', methods=['POST'])
def crop_image():
    # Load the corrected image from the processed folder
    image_path = os.path.join(app.config['PROCESSED_FOLDER'], "corrected.png")
    if not os.path.exists(image_path):
        return jsonify({"error": "Corrected image not found"}), 404

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Failed to load corrected image"}), 400

    # Convert to grayscale and apply a threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    thresh_inv = cv2.bitwise_not(thresh)

    # Detect horizontal lines
    kernel = np.ones((1, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)

    # Find contours after removing lines
    contours, _ = cv2.findContours(thresh_inv - horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop the image and save it if contours are found
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y + h, x:x + w]

        # Save the cropped image
        cropped_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'cropped_image.png')
        cv2.imwrite(cropped_image_path, cropped_image)

        # Encode the cropped image to base64 for response
        _, buffer = cv2.imencode('.png', cropped_image)
        cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "cropped_image": f"data:image/png;base64,{cropped_image_base64}",
            "cropped_image_url": url_for('send_processed_image', filename='cropped_image.png', _external=True)
        })
    else:
        return jsonify({"error": "No contours found to crop the image"}), 400

@app.route('/static/processed/<filename>')
def send_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
