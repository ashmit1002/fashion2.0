import io
import os
import requests
from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import vision
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Initialize Flask app and Google Vision client
app = Flask(__name__)
client = vision.ImageAnnotatorClient()

# Function to get the dominant color from an image
def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].round().astype(int)
    return tuple(dominant_color)

# Function to get the closest color name from TheColorAPI
def get_color_name(rgb):
    url = "https://www.thecolorapi.com/id"
    response = requests.get(url, params={'rgb': f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'})
    if response.status_code == 200:
        color_data = response.json()
        color_name = color_data['name']['value']
        return color_name
    return None

# Function to process image, detect objects, and visualize with bounding boxes
def detect_objects(image_data):
    image = vision.Image(content=image_data)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    descriptions = []
    
    # Loop through the detected objects and draw bounding boxes
    for obj in objects:
        box = obj.bounding_poly.normalized_vertices
        vertices = [(int(vertex.x * img.shape[1]), int(vertex.y * img.shape[0])) for vertex in box]
        x_min, y_min = vertices[0]
        x_max, y_max = vertices[2]
        cropped_region = img[y_min:y_max, x_min:x_max]
        dominant_color = get_dominant_color(cropped_region)
        color_name = get_color_name(dominant_color)
        dominant_color_bgr = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))

        # Draw bounding box on the image
        cv2.polylines(img, [np.array(vertices)], isClosed=True, color=dominant_color_bgr, thickness=2)
        label = obj.name
        cv2.putText(img, f"{label} - {color_name}", (vertices[0][0], vertices[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dominant_color_bgr, 2)

        # Append the description to the result
        descriptions.append(f"{label} - {color_name}")
    
    # Save the processed image
    processed_image_path = "/tmp/processed_image.jpg"
    cv2.imwrite(processed_image_path, img)

    return processed_image_path, descriptions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_data = file.read()
    processed_image_path, descriptions = detect_objects(image_data)

    # Send the processed image and descriptions to frontend
    return jsonify({'image_path': processed_image_path, 'descriptions': descriptions})

@app.route('/processed_image')
def serve_processed_image():
    image_path = request.args.get('path')
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
