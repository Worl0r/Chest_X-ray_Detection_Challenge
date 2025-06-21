import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torchvision.ops import box_iou
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_augmentation import load_train_dataset, create_transforms, perform_data_augmentation, ChestXrayDataset, encode_targets_for_detection, collate_fn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

def get_fasterrcnn_model(num_classes):
    # num_classes = nombre total de classes (y compris le fond)
    # ici classes + background => background = 0
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Remplacer la tête de classification par une nouvelle adaptée à notre nb de classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# === Boucle d'entraînement ===
def train(model, train_loader, val_loader, optimizer, scheduler, device, patience, epochs=5):
    model.to(device)
    best_auc = 0

    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir="runs/fasterrcnn_experiment")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k,v in elem.items()} for elem in targets]

            optimizer.zero_grad()

            # In training mode, with targets provided, this returns a dict of losses
            loss_dict = model(images, targets)
            loss = sum(lo for lo in loss_dict.values())

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        
        # Perform validation ONCE per epoch
        mAP, mean_precision, mean_recall, mean_f1 = validate(model, val_loader, device, classes=classes)
        
        # Update scheduler based on validation results
        scheduler.step(mAP)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val mAP: {mAP:.4f}")

        writer.add_scalar("Metrics/mAP", mAP, epoch)
        writer.add_scalar("Metrics/Precision", mean_precision, epoch)
        writer.add_scalar("Metrics/Recall", mean_recall, epoch)
        writer.add_scalar("Metrics/F1", mean_f1, epoch)

        if mAP > best_auc:
            best_auc = mAP
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model saved.")

        early_stopping(mAP)
        if early_stopping.early_stop:
            print("⏹️ Early stopping triggered")
            break

    writer.close()
    print(f"✅ Training complete. Best AUC: {best_auc:.4f}")


# === Évaluation ===

def validate(model, dataloader, device, iou_threshold=0.5, score_threshold=0.1, classes=None):
    model.eval()
    
    # For each class, store detections sorted by confidence
    class_detections = defaultdict(list)
    # Store ground truth counts per class
    class_gt_counts = defaultdict(int)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            
            outputs = model(images)
            
            for i in range(len(images)):
                preds = outputs[i]
                gt = targets[i]
                
                pred_boxes = preds['boxes'].cpu()
                pred_scores = preds['scores'].cpu()
                pred_labels = preds['labels'].cpu()
                
                true_boxes = gt['boxes'].cpu()
                true_labels = gt['labels'].cpu()
                
                # Count ground truths for each class
                for label in true_labels:
                    class_gt_counts[label.item()] += 1
                
                # Store all detections by class with their confidence scores
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score >= score_threshold:
                        class_detections[label.item()].append({
                            'box': box,
                            'score': score,
                            'matched': False,  # Will track if this detection matched a GT
                            'image_idx': i
                        })
                        
                # Store GT boxes for each image and class
                for j, (box, label) in enumerate(zip(true_boxes, true_labels)):
                    cls = label.item()
                    if 'gt_boxes' not in class_detections[cls]:
                        class_detections[cls].append({'gt_boxes': defaultdict(list)})
                    
                    class_detections[cls][-1]['gt_boxes'][i].append({
                        'box': box,
                        'matched': False  # Will track if this GT matched a detection
                    })
    
    # Calculate AP for each class
    APs = []
    precisions = []
    recalls = []
    f1s = []
    
    # For each class
    for cls in class_detections:
        if cls == 0:  # Skip background class
            continue
            
        # Skip if no ground truths
        if class_gt_counts[cls] == 0:
            continue
            
        # Extract detections for this class
        dets = [d for d in class_detections[cls] if 'box' in d]
        
        # Sort by decreasing confidence
        dets.sort(key=lambda x: x['score'], reverse=True)
        
        # Extract GT boxes
        gt_dict = next((item['gt_boxes'] for item in class_detections[cls] if 'gt_boxes' in item), defaultdict(list))
        
        # Compute precision-recall curve
        tp = 0
        fp = 0
        precision_values = []
        recall_values = []
        
        for det in dets:
            # Get image index and detection box
            img_idx = det['image_idx']
            d_box = det['box'].unsqueeze(0)
            
            # Get GT boxes for this image
            gt_boxes_img = gt_dict[img_idx]
            
            # If no GT boxes in this image, it's a false positive
            if not gt_boxes_img:
                fp += 1
            else:
                # Calculate IoU with all GT boxes
                gt_boxes_tensor = torch.stack([g['box'] for g in gt_boxes_img])
                ious = box_iou(d_box, gt_boxes_tensor)[0]
                
                # Check if there's a match
                if torch.any(ious >= iou_threshold):
                    best_match_idx = torch.argmax(ious).item()
                    
                    # If this GT hasn't been matched yet
                    if not gt_boxes_img[best_match_idx]['matched']:
                        tp += 1
                        gt_boxes_img[best_match_idx]['matched'] = True
                    else:
                        fp += 1  # Already matched, so this is a duplicate detection
                else:
                    fp += 1  # No matching GT box
            
            # Calculate precision and recall at this point
            precision = tp / (tp + fp)
            recall = tp / class_gt_counts[cls]
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        # Compute AP using precision-recall curve
        # Convert to numpy for easier interpolation
        precision_values = np.array(precision_values)
        recall_values = np.array(recall_values)
        
        # Ensure precision values are non-decreasing when going from right to left
        for i in range(len(precision_values)-2, -1, -1):
            precision_values[i] = max(precision_values[i], precision_values[i+1])
        
        # Add sentinel values
        precision_values = np.concatenate(([0], precision_values, [0]))
        recall_values = np.concatenate(([0], recall_values, [1]))
        
        # Compute AP as area under precision-recall curve
        # Find points where recall changes
        recall_changes = np.where(recall_values[1:] != recall_values[:-1])[0]
        
        # Calculate AP using precision at each recall change point
        AP = np.sum((recall_values[recall_changes + 1] - recall_values[recall_changes]) * 
                    precision_values[recall_changes + 1])
        
        APs.append(AP)
        
        # Calculate final precision, recall, F1 based on all detections
        final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        final_recall = tp / class_gt_counts[cls] if class_gt_counts[cls] > 0 else 0
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
        
        precisions.append(final_precision)
        recalls.append(final_recall)
        f1s.append(final_f1)
        
        print(f"Class {classes[cls-1] if classes else cls}: AP={AP:.3f}, Precision={final_precision:.3f}, Recall={final_recall:.3f}, F1={final_f1:.3f}")
    
    # Calculate mAP
    mAP = sum(APs) / len(APs) if len(APs) > 0 else 0
    mean_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    mean_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    mean_f1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0
    
    print(f"\nMean Average Precision (mAP) @ IoU={iou_threshold}: {mAP:.3f}")
    print(f"Mean Precision: {mean_precision:.3f}")
    print(f"Mean Recall: {mean_recall:.3f}")
    print(f"Mean F1-score: {mean_f1:.3f}")
    
    return mAP, mean_precision, mean_recall, mean_f1

def freeze_backbone(model):
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad = False

def unfreeze_backbone(model):
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad = True

def gradual_unfreeze(model, steps=3):
    layers = list(model.backbone.body.children())
    total_layers = len(layers)
    for i in range(total_layers):
        for param in layers[i].parameters():
            param.requires_grad = (i >= total_layers - steps)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


# === Main ===
if __name__ == "__main__":
    CSV_PATH = "train.csv"
    IMAGE_DIR = "train"
    BATCH_SIZE = 4
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-4
    weight_decay = 1e-2
    # gamma = 0.95

    df = load_train_dataset()

    df = perform_data_augmentation(df)

    train_transform = create_transforms()

    all_labels = [label for labels in df["annotation"].apply(lambda x: x["labels"]) for label in labels]
    classes = sorted(set(all_labels))
    class_to_idx = {cls: i + 1 for i, cls in enumerate(classes)}
    y_encoded = encode_targets_for_detection(df, class_to_idx)


    dataset = ChestXrayDataset(df, y_encoded, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = get_fasterrcnn_model(len(classes)+1)

    # Start with frozen backbone
    freeze_backbone(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # First training phase with frozen backbone
    print("Phase 1: Training with frozen backbone")
    train(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=3, patience=5)
    
    # Gradual unfreezing in multiple steps
    total_layers = len(list(model.backbone.body.children()))
    unfreeze_steps = 3  # Number of gradual unfreezing steps
    
    for step in range(1, unfreeze_steps + 1):
        print(f"Phase {step+1}: Gradually unfreezing, step {step}/{unfreeze_steps}")
        
        # Calculate how many layers to unfreeze in this step
        layers_to_unfreeze = (step * total_layers) // unfreeze_steps
        
        # Apply gradual unfreezing
        gradual_unfreeze(model, steps=layers_to_unfreeze)
        
        # Update optimizer to include newly unfrozen parameters
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate * (0.5 ** step), weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        # Train for a few epochs with current unfreezing
        train(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=2, patience=3)
    
    # Final training phase with fully unfrozen model
    print("Final phase: Training with fully unfrozen model")
    unfreeze_backbone(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.1, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    train(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=3, patience=5)

