from flask import Flask, request, jsonify, url_for, send_from_directory
import cv2
import numpy as np
import os
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

# Define image processing functions
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.equalizeHist(image)

def calculate_skew_angle(image):
    points = np.column_stack(np.where(image > 0))
    if points.shape[0] < 2:
        raise ValueError("Not enough points to calculate PCA.")
    mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No filename provided"}), 400

    # Save original image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Process the image: Gaussian blur, edge detection, and contour finding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mask the contours to isolate the bone
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    isolated_bone = cv2.bitwise_and(image, image, mask=mask)

    # Calculate skew angle and correct it
    try:
        skew_angle = calculate_skew_angle(isolated_bone)
        corrected_image = rotate_image(isolated_bone, -skew_angle)
        equalized_image = preprocess_image(image_path)  # Apply histogram equalization to the original image

        # Save processed images
        isolated_bone_path = os.path.join(app.config['PROCESSED_FOLDER'], 'isolated_bone.png')
        corrected_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'corrected.png')
        equalized_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'equalized.png')
        cv2.imwrite(isolated_bone_path, isolated_bone)
        cv2.imwrite(corrected_image_path, corrected_image)
        cv2.imwrite(equalized_image_path, equalized_image)

        # Return URLs for the images
        return jsonify({
            "isolated": url_for('send_processed_image', filename='isolated_bone.png', _external=True),
            "corrected": url_for('send_processed_image', filename='corrected.png', _external=True),
            "equalized": url_for('send_processed_image', filename='equalized.png', _external=True)
        })

    except ValueError as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 400

@app.route('/static/processed/<filename>')
def send_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
