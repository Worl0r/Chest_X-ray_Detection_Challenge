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

    writer = SummaryWriter(log_dir="runs/fasterrcnn_experiment")  # Création du writer


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k,v in elem.items()} for elem in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()

            scheduler.step()

            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader.dataset) + i)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)


        mAP, mean_precision, mean_recall, mean_f1 = validate(model, val_loader, device)

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

    writer.close()  # Fermer le writer
    print(f"✅ Training complete. Best AUC: {best_auc:.4f}")


# === Évaluation ===

def validate(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5, classes=None):
    model.eval()

    all_preds = []
    all_gts = []
    all_pred_labels = []
    all_true_labels = []

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

                # Filtrer par score threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                # Garder aussi les vraies étiquettes
                all_gts.append(true_boxes)
                all_true_labels.append(true_labels)

                all_preds.append(pred_boxes)
                all_pred_labels.append(pred_labels)

    # Concaténer tout
    all_preds = torch.cat(all_preds) if len(all_preds) > 0 else torch.empty((0,4))
    all_pred_labels = torch.cat(all_pred_labels) if len(all_pred_labels) > 0 else torch.empty((0,), dtype=torch.int64)
    all_gts = torch.cat(all_gts) if len(all_gts) > 0 else torch.empty((0,4))
    all_true_labels = torch.cat(all_true_labels) if len(all_true_labels) > 0 else torch.empty((0,), dtype=torch.int64)

    # Pour chaque prédiction on va chercher un vrai positif
    TP, FP, FN = 0, 0, 0
    matched_gt_indices = set()

    # On calcule pour chaque classe séparément
    class_metrics = defaultdict(lambda: {"TP":0, "FP":0, "FN":0})

    max_true = all_true_labels.max().item() if all_true_labels.numel() > 0 else 0
    max_pred = all_pred_labels.max().item() if all_pred_labels.numel() > 0 else 0

    max_cls = max(max_true, max_pred)

    for cls in range(1, max_cls + 1):
        pred_mask = (all_pred_labels == cls)
        gt_mask = (all_true_labels == cls)

        pred_boxes_cls = all_preds[pred_mask]
        gt_boxes_cls = all_gts[gt_mask]

        matched = set()

        for i, pbox in enumerate(pred_boxes_cls):
            if len(gt_boxes_cls) == 0:
                # Pas de vrai bbox, c'est un FP
                class_metrics[cls]["FP"] += 1
                continue

            ious = box_iou(pbox.unsqueeze(0), gt_boxes_cls)[0]
            max_iou, max_idx = torch.max(ious, dim=0)

            if max_iou >= iou_threshold and max_idx.item() not in matched:
                class_metrics[cls]["TP"] += 1
                matched.add(max_idx.item())
            else:
                class_metrics[cls]["FP"] += 1

        # Les GT non appariés sont des FN
        FN_cls = len(gt_boxes_cls) - len(matched)
        class_metrics[cls]["FN"] += FN_cls

    # Calcul global mAP simple (AP @ IoU=0.5) comme moyenne des précisions par classe
    APs = []
    precisions = []
    recalls = []
    f1s = []

    for cls, met in class_metrics.items():
        TP = met["TP"]
        FP = met["FP"]
        FN = met["FN"]

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        APs.append(precision)  # approximation simple pour AP = precision à ce seuil
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"Classe {classes[cls-1] if classes else cls}: TP={TP}, FP={FP}, FN={FN}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    mAP = sum(APs) / len(APs) if len(APs) > 0 else 0.0
    mean_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0.0
    mean_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0.0
    mean_f1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0.0

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
    BATCH_SIZE = 1
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-4
    weight_decay = 1e-2
    gamma = 0.95

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

    freeze_backbone(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    train(model, train_loader, val_loader, optimizer, scheduler,DEVICE, epochs=5, patience=5)

    unfreeze_backbone(model)

    train(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=EPOCHS, patience=5)


