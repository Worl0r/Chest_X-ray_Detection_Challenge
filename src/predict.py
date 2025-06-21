import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import csv
from collections import defaultdict

def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes, device):
    model = get_fasterrcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image), image.size

def predict(model, image_tensor, confidence_threshold=0.5, device="cpu"):
    """
    faut mettre le confidence_threshold a 0, car sinon on a des images pour les quelles on ne fait pas de prediction
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
    
    # Extract predictions above threshold
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels

def main():
    # Configuration
    MODEL_PATH = "best_model.pth"
    TEST_DIR = "test"
    MAPPING_CSV = "ID_to_Image_Mapping.csv"
    OUTPUT_CSV = "predictions/predictions_7.csv"
    CONFIDENCE_THRESHOLD = 0
    
    # Classes that were used during training (8 classes + background)
    all_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
                  "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
    classes = sorted(set(all_labels))
    
    print(f"Using {len(classes)} classes: {classes}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    num_classes = len(classes) + 1  # +1 for background class
    model = load_model(MODEL_PATH, num_classes, device)
    print("Model loaded successfully.")
    
    # Load image mappings
    mapping_df = pd.read_csv(MAPPING_CSV)
    print(f"Loaded {len(mapping_df)} image mappings.")
    
    # Dictionary to track how many times each image has appeared
    image_appearance_count = defaultdict(int)
    
    # Cache for storing predictions for repeated images
    image_predictions_cache = {}
    
    # Open output file
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'label'])
        
        # Process each image in the mapping
        for _, row in mapping_df.iterrows():
            id_num = row['id']
            image_id = row['image_id']
            image_path = os.path.join(TEST_DIR, image_id)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
            
            # Track occurrences of this image
            image_appearance_count[image_id] += 1
            current_appearance = image_appearance_count[image_id]
            
            # Check if we've already processed this image
            if image_id in image_predictions_cache:
                cached_boxes, cached_scores, cached_labels = image_predictions_cache[image_id]
            else:
                # Preprocess image
                image_tensor, original_size = preprocess_image(image_path)
                
                # Get predictions
                boxes, scores, labels = predict(model, image_tensor, CONFIDENCE_THRESHOLD, device)
                
                # Sort predictions by confidence score (highest first)
                if len(boxes) > 0:
                    sort_indices = np.argsort(scores)[::-1]  # Sort in descending order
                    boxes = boxes[sort_indices]
                    scores = scores[sort_indices]
                    labels = labels[sort_indices]
                
                # Cache the predictions
                image_predictions_cache[image_id] = (boxes, scores, labels)
                
                cached_boxes, cached_scores, cached_labels = boxes, scores, labels
            
            # If no predictions above threshold, continue to next image
            if len(cached_boxes) == 0:
                print(f"No predictions above threshold for image {image_id}")
                continue
            
            # Determine which prediction to use based on appearance count
            prediction_index = min(current_appearance - 1, len(cached_boxes) - 1)
            
            # Extract the prediction to use
            box = cached_boxes[prediction_index:prediction_index+1]
            score = cached_scores[prediction_index:prediction_index+1]
            label = cached_labels[prediction_index:prediction_index+1]
            
            # Write prediction to CSV
            for b, s, l in zip(box, score, label):
                # Convert label index to class name (subtract 1 as index 0 is background)
                class_name = classes[l-1] if l > 0 and l <= len(classes) else "Unknown"
                
                # Write row: id,image_id,x_min,y_min,x_max,y_max,confidence,label
                writer.writerow([
                    id_num,
                    image_id,
                    int(b[0]),
                    int(b[1]),
                    int(b[2]),
                    int(b[3]),
                    float(s),
                    class_name
                ])
            
            print(f"Processed image {image_id} (ID: {id_num}) - Appearance #{current_appearance}")
    
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()