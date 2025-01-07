import io
import os
import requests
import base64
from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import vision
import cv2
import numpy as np
from sklearn.cluster import KMeans
from serpapi import GoogleSearch

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

# Function to perform reverse image search using SerpAPI (Google Image Search API)
def reverse_image_search(cropped_image):
    # Convert the cropped image to a byte array
    _, img_encoded = cv2.imencode('.jpg', cropped_image)
    img_byte_array = img_encoded.tobytes()

    # Convert image to base64
    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

    # SerpAPI parameters
    params = {
        "q": "image search",
        "tbm": "isch",  # Image Search
        "ijn": "0",
        "api_key": "ef1060959fb01ad0a7d0000ed737a785872acf6d6b17b12ee71ef7b575e88999",  # Your SerpAPI API key
        "encoded_image": img_base64  # Base64 encoded image
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Check for results and extract data
    if 'images_results' in results:
        image_info = results['images_results'][0]  # Get the first match
        item_url = image_info.get('original', 'N/A')
        title = image_info.get('title', 'N/A')
        price = image_info.get('source', 'N/A')  # You can extract price if available in metadata
        return item_url, title, price

    return None, None, None

# Function to process image, detect objects, and visualize with bounding boxes
def detect_objects(image_data):
    image = vision.Image(content=image_data)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    descriptions = []
    clothing_info = []

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

        # Perform reverse image search for clothing items
        item_url, title, price = reverse_image_search(cropped_region)
        if item_url:
            clothing_info.append({
                'label': label,
                'color': color_name,
                'item_url': item_url,
                'title': title,
                'price': price
            })

    # Save the processed image
    processed_image_path = "/tmp/processed_image.jpg"
    cv2.imwrite(processed_image_path, img)

    return processed_image_path, descriptions, clothing_info

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
    processed_image_path, descriptions, clothing_info = detect_objects(image_data)

    # Send the processed image and descriptions to frontend
    return jsonify({
        'image_path': processed_image_path,
        'descriptions': descriptions,
        'clothing_info': clothing_info
    })

@app.route('/processed_image')
def serve_processed_image():
    image_path = request.args.get('path')
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
