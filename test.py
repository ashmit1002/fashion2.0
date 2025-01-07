import os
import io
import requests
from google.cloud import vision
import cv2
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, send_file, render_template

# Initialize Flask app
app = Flask(__name__)

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Function to get the dominant color from an image
def get_dominant_color(image):
    i = 0
    j= 5
    k = i + j
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

# Route for home page to check if server is running
@app.route('/')
def home():
    return render_template('upload.html')  # Serve the HTML form

# Route to handle image upload and analysis
# Route to handle image upload and analysis
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get the image file from the request
    image_file = request.files['image']
    
    # Ensure the image has a filename before saving
    if not image_file.filename:
        return jsonify({'error': 'No filename found in the uploaded file'}), 400
    
    # Save the uploaded image
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    # Analyze the image (bounding boxes, dominant colors, etc.)
    try:
        img = cv2.imread(image_path)
        with io.open(image_path, 'rb') as f:
            content = f.read()
        image = vision.Image(content=content)

        # Perform object detection
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations

        components = []
        annotated_image_path = 'static/annotated_' + image_file.filename

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

            # Store component data
            components.append({
                'name': label,
                'dominant_color': color_name,
                'image_url': f'/static/{image_file.filename}'  # Display the original image thumbnail
            })

        # Save the annotated image
        cv2.imwrite(annotated_image_path, annotated_image)

        return jsonify({
            'components': components,
            'annotated_image_url': f'/static/annotated_{image_file.filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)




# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Fashion App - Outfit Analyzer</title>
#     <style>
#         #output-container {
#             margin-top: 20px;
#         }
#         .component {
#             margin-bottom: 10px;
#         }
#         #annotated-image {
#             margin-top: 20px;
#             max-width: 100%;
#             height: auto;
#         }
#     </style>
#     <script>
#         async function uploadImage() {
#             const formData = new FormData();
#             const imageFile = document.getElementById("image-file").files[0];
#             formData.append("image", imageFile);

#             const response = await fetch('/upload', {
#                 method: 'POST',
#                 body: formData
#             });

#             const data = await response.json();
            
#             if (data.error) {
#                 alert("Error: " + data.error);
#                 return;
#             }

#             // Display the detected components and their dominant colors
#             const componentsContainer = document.getElementById("components-container");
#             componentsContainer.innerHTML = ''; // Clear previous results

#             data.components.forEach(component => {
#                 const componentDiv = document.createElement("div");
#                 componentDiv.classList.add("component");
#                 componentDiv.innerHTML = `
#                     <strong>${component.name}</strong><br>
#                     <span>Color: ${component.dominant_color}</span><br>
#                     <img src="${component.image_url}" alt="${component.name}" style="max-width: 100px;">
#                 `;
#                 componentsContainer.appendChild(componentDiv);
#             });

#             // Display the image with bounding boxes
#             const annotatedImage = document.getElementById("annotated-image");
#             annotatedImage.src = data.annotated_image_url;
#         }
#     </script>
# </head>
# <body>

#     <h1>Fashion App - Outfit Analyzer</h1>

#     <label for="image-file">Upload an Image:</label>
#     <input type="file" id="image-file" accept="image/*">
#     <button onclick="uploadImage()">Upload & Analyze</button>

#     <div id="output-container">
#         <div id="components-container"></div>
#         <br>
#         <h3>Image with Bounding Box Annotations:</h3>
#         <img id="annotated-image" src="" alt="Annotated Image">
#     </div>

# </body>
# </html>
