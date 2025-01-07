import io
import requests
from google.cloud import vision
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Function to get the dominant color from an image
def get_dominant_color(image):
    # Convert image to RGB (if it's not in RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels (each pixel as a row)
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans clustering to find the dominant color
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)  # Explicitly setting n_init to suppress the warning
    kmeans.fit(pixels)
    
    # Get the center of the cluster (dominant color)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Convert the dominant color from float to integer
    dominant_color = dominant_color.round().astype(int)
    
    return tuple(dominant_color)

# Function to get the closest color name from TheColorAPI
def get_color_name(rgb):
    # TheColorAPI endpoint
    url = "https://www.thecolorapi.com/id"
    
    # Send a GET request to get the color name for the RGB value
    response = requests.get(url, params={'rgb': f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'})
    
    # If the request was successful, extract the color name
    if response.status_code == 200:
        color_data = response.json()
        color_name = color_data['name']['value']
        return color_name
    else:
        return None

# Function to detect objects, extract bounding boxes, and visualize with dominant color
def detect_objects(image_path):
    # Load the image
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Load the image with OpenCV for visualization
    img = cv2.imread(image_path)

    # Loop through the detected objects and draw bounding boxes with dominant color
    for obj in objects:
        # Get the bounding box coordinates
        box = obj.bounding_poly.normalized_vertices
        vertices = [(int(vertex.x * img.shape[1]), int(vertex.y * img.shape[0])) for vertex in box]

        # Crop the image to the bounding box area
        x_min, y_min = vertices[0]
        x_max, y_max = vertices[2]
        cropped_region = img[y_min:y_max, x_min:x_max]

        # Get the dominant color for the cropped region
        dominant_color = get_dominant_color(cropped_region)

        # Get the color name using TheColorAPI
        color_name = get_color_name(dominant_color)

        # Ensure dominant_color is in the correct format for OpenCV (BGR)
        dominant_color_bgr = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))  # Convert RGB to BGR

        # Draw the bounding box on the image with the dominant color
        cv2.polylines(img, [np.array(vertices)], isClosed=True, color=dominant_color_bgr, thickness=2)

        # Optionally: Add the label name and color name of the object
        label = obj.name
        cv2.putText(img, f"{label} - {color_name}", (vertices[0][0], vertices[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dominant_color_bgr, 2)

    # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes and Dominant Colors", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your image
image_path = '/Users/ashmi/Downloads/Coding/Fahion App/anfModelTest copy.jpeg'

# Detect objects, extract bounding boxes, and visualize with dominant colors
detect_objects(image_path)
