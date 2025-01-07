import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

# Set up Detectron2 config
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'  # Use CPU instead of CUDA
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # Set threshold for detection

# Initialize the model
predictor = DefaultPredictor(cfg)

# Load your image
image_path = "/Users/ashmi/Downloads/Coding/Fahion App/anfModelTest copy.jpeg"
image = cv2.imread(image_path)

# Run Detectron2 on the image
outputs = predictor(image)

# Get the bounding boxes and masks
instances = outputs["instances"]
boxes = instances.pred_boxes.tensor.cpu().numpy()
masks = instances.pred_masks.cpu().numpy()

# Visualize segmentation masks for clothing
for i in range(len(boxes)):
    box = boxes[i]
    mask = masks[i]
    
    # Convert mask to binary
    mask = mask.astype(np.uint8) * 255
    
    # Apply the mask to the image
    segmented_clothing = cv2.bitwise_and(image, image, mask=mask)
    
    # Draw bounding box
    cv2.rectangle(image, 
                  (int(box[0]), int(box[1])), 
                  (int(box[2]), int(box[3])), 
                  (0, 255, 0), 2)
    
    # Show segmented clothing
    cv2.imshow("Segmented Clothing", segmented_clothing)
    cv2.waitKey(0)

# Show the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
