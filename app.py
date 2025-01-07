import os
import io
import requests
import boto3
from google.cloud import vision
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from botocore.exceptions import NoCredentialsError
from serpapi import GoogleSearch
from sklearn.cluster import KMeans
import base64

# Initialize Flask app
app = Flask(__name__)

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Initialize AWS S3 client
s3_client = boto3.client('s3')

# Set your S3 bucket name
S3_BUCKET_NAME = 'fashionwebapp'

# Function to get the dominant color from an image
def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    dominant_color = dominant_color.round().astype(int)
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

# Function to upload an image to AWS S3
def upload_to_s3(image, filename):
    try:
        # Convert image to bytes
        _, img_bytes = cv2.imencode('.jpg', image)
        img_bytes = img_bytes.tobytes()

        # Upload image to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=img_bytes,
            ContentType='image/jpeg'
        )

        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
        return s3_url

    except NoCredentialsError:
        return None

# Function to get clothing items from Google Image Search
def get_clothing_from_google_search(image_url):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": "ef1060959fb01ad0a7d0000ed737a785872acf6d6b17b12ee71ef7b575e88999"
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        visual_matches = results.get("visual_matches", [])
        top_matches = visual_matches[:3]

        response = []
        for match in top_matches:
            response.append({
                "title": match.get("title"),
                "link": match.get("link"),
                "price": str(match.get("price", {}).get("currency", "")) + " " + str(match.get("price", {}).get("extracted_value", "N/A")),
                "thumbnail": match.get("thumbnail")
            })

        return response

    except Exception as e:
        return {"error": str(e)}

# Default route to serve the frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Route to handle image upload and analysis
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    image_filename = image_file.filename  # Get the filename from the file object
    image_path = os.path.join('uploads', image_filename)

    # Save the uploaded image
    image_file.save(image_path)

    # Analyze the image (bounding boxes, dominant colors, etc.)
    try:
        img = cv2.imread(image_path)
        
        # Open the saved image for Google Vision API
        with io.open(image_path, 'rb') as img_file:
            content = img_file.read()
        image = vision.Image(content=content)

        # Perform object detection
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations

        components = []
        annotated_image_path = os.path.join('static', 'annotated_' + image_filename)

        # Create a copy of the original image for annotating
        annotated_image = img.copy()

        for obj in objects:
            box = obj.bounding_poly.normalized_vertices
            vertices = [(int(vertex.x * img.shape[1]), int(vertex.y * img.shape[0])) for vertex in box]

            # Get the dominant color for the bounding box area
            x_min, y_min = vertices[0]
            x_max, y_max = vertices[2]
            cropped_region = img[y_min:y_max, x_min:x_max]
            dominant_color = get_dominant_color(cropped_region)

            # Get the color name
            color_name = get_color_name(dominant_color)

            # Draw bounding boxes and labels
            dominant_color_bgr = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
            cv2.polylines(annotated_image, [np.array(vertices)], isClosed=True, color=dominant_color_bgr, thickness=2)
            label = obj.name
            cv2.putText(annotated_image, f"{label} - {color_name}", (vertices[0][0], vertices[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dominant_color_bgr, 2)

            # Save the bounding box as a separate image
            cropped_image = img[y_min:y_max, x_min:x_max]
            filename = f"{label}_{x_min}_{y_min}.jpg"
            image_url = upload_to_s3(cropped_image, filename)

            if image_url:
                # Get clothing item details from Google Search
                clothing_items = get_clothing_from_google_search(image_url)

                # Append component data
                components.append({
                    'name': label,
                    'dominant_color': color_name,
                    'image_url': image_url,
                    'clothing_items': clothing_items
                })

        # Convert annotated image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'components': components,
            'annotated_image_base64': annotated_image_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure required directories exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')

    # Run the Flask app
    app.run(debug=True)
