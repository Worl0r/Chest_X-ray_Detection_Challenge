import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import MultiLabelBinarizer

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch


import matplotlib.pyplot as plt

def show_image_with_bboxes(images, bboxes, labels, classes):
    """
    Affiche une image avec ses bounding boxes et labels (format coco : [x, y, w, h] en pixels).
    """
    n = len(images)
    plt.figure(figsize=(16, 8))

    for i in range(n):
        img_i = images[i]
        bboxes_i = bboxes[i]
        labels_i = labels[i]

        # Convertir Tensor en NumPy
        if isinstance(img_i, torch.Tensor):
            img_i = img_i.permute(1, 2, 0).cpu().numpy()

        # Si float [0,1], convertir en uint8
        if img_i.dtype in [np.float32, np.float64]:
            img_i = (img_i * 255).clip(0, 255).astype(np.uint8)

        # Si 1 canal → RGB
        if img_i.ndim == 2:
            img_i = cv2.cvtColor(img_i, cv2.COLOR_GRAY2RGB)
        elif img_i.shape[2] == 1:
            img_i = cv2.cvtColor(img_i, cv2.COLOR_GRAY2RGB)

        img_i = np.ascontiguousarray(img_i)

        # Dessiner chaque bbox
        for bbox, label in zip(bboxes_i, labels_i):
            if isinstance(label, torch.Tensor):
                label = label.item()
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)  # Vert

            cv2.rectangle(img_i, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_i, classes[int(label)-1], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, lineType=cv2.LINE_AA)

        plt.subplot(2, (n + 1) // 2, i + 1)
        plt.imshow(img_i)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def list_files_in_folder(folder_path, extensions=None):
    """
    Liste tous les fichiers dans un dossier, avec une option pour filtrer par extensions.

    Args:
        folder_path (str): Chemin du dossier.
        extensions (tuple[str], optional): Extensions à garder, ex: ('.jpg', '.png'). Si None, garde tout.

    Returns:
        List[str]: Liste des chemins des fichiers.
    """
    files = []
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if extensions is None or f.lower().endswith(extensions):
                files.append(f)
    return files

# === 1. Charger les labels & bboxes ===

def load_train_dataset():
    df_annots = pd.read_csv("train.csv")  # colonnes: Image Index, Finding Label, Bbox [x, y, w, h]

    df_grouped = df_annots.groupby("Image Index").apply(lambda g: {
        "labels": list(g["Finding Label"]),
        "bboxes": g[["Bbox [x", "y", "w", "h]"]].values.tolist()
    }).reset_index(name="annotation")

    df = df_grouped
    # df = df_labels.merge(df_grouped, left_on="image_id", right_on="Image Index", how="left")
    df["annotation"] = df["annotation"].apply(lambda x: x if isinstance(x, dict) else {"labels": [], "bboxes": []})

    return df

def perform_data_augmentation(df, min_count = 5):
    all_labels = [label for labels in df["annotation"].apply(lambda x: x["labels"]) for label in labels]
    counts = Counter(all_labels)

    aug_rows = []
    for cls, count in counts.items():
        if count < min_count:
            subset = df[df["annotation"].apply(lambda x: cls in x["labels"])]
            needed = min_count - count
            resampled = resample(subset, replace=True, n_samples=needed, random_state=42)
            aug_rows.append(resampled)

    if aug_rows:
        df_aug = pd.concat(aug_rows)
        df = pd.concat([df, df_aug])

    df = df.sample(frac=1).reset_index(drop=True)

    return df


def encode_targets_for_detection(df, class_to_idx):
    targets = []
    for ann in df["annotation"]:
        boxes = []
        labels = []
        for bbox, label in zip(ann["bboxes"], ann["labels"]):
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # [xmin, ymin, xmax, ymax]
            labels.append(class_to_idx[label])
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })
    return targets


def create_transforms():
    train_transform = A.Compose([
        A.Resize(512, 512),
        # A.HorizontalFlip(p=0.5),
        # A.Affine(rotate=(-5, 5), translate_percent=(0.05, 0.05), scale=(0.95, 1.05), p=0.7),
        # A.RandomBrightnessContrast(p=0.1),
        # A.GaussNoise(p=0.1),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["class_labels"]))

    return train_transform


class ChestXrayDataset(Dataset):
    def __init__(self, df, targets, transform=None):
        self.df = df
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join("train", row["Image Index"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        target = self.targets[idx]
        bboxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            # On récupère les bboxes mises à jour
            bboxes = transformed["bboxes"]

        # Attention : ici on doit reconstruire le `target` sous forme torch.Tensor
        final_target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, final_target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets



# === 7. Test ===
if __name__ == "__main__":
    df = load_train_dataset()
    # df = perform_data_augmentation(df)

    # Obtenir classes et mapping
    all_labels = [label for labels in df["annotation"].apply(lambda x: x["labels"]) for label in labels]
    classes = sorted(set(all_labels))
    class_to_idx = {cls: i + 1 for i, cls in enumerate(classes)}

    targets = encode_targets_for_detection(df, class_to_idx)

    transform = create_transforms()

    dataset = ChestXrayDataset(df, targets, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    images, targets = next(iter(loader))
    print("Image batch size:", len(images))
    print("First image shape:", images[0].shape)
    print("First target:", targets[0])

    show_image_with_bboxes(images, [item["boxes"] for item in targets], [item["labels"] for item in targets], classes)

    print("end")
