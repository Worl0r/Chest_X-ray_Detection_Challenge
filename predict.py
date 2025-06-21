import os
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

from torchvision.ops import nms


# === Define Transforms ===
def create_transforms():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


# === Dataset for Test ===
class ChestXrayTestDataset(Dataset):
    def __init__(self, image_dir, image_filenames, transform=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_id = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, image_id


# === Load Model ===
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# === Inference ===
def run_inference(model, dataloader, classes, device, mapping_df, score_thresh=0.5):
    results = []
    global_id = 0

    with torch.no_grad():
        for images, image_ids in tqdm(dataloader, desc="Predicting"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for image_id, output in zip(image_ids, outputs):
                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                if scores.numel() == 0:
                    continue  # pas de prédiction

                top_idx = scores.argmax()
                top_score = scores[top_idx].item()

                # if top_score < score_thresh:
                #     continue  # score trop faible

                top_box = boxes[top_idx].cpu().numpy()
                top_label = labels[top_idx].item()

                x_min, y_min, x_max, y_max = top_box
                label_name = classes[top_label - 1]  # attention à l’indexation

                results.append(
                    {
                        "id": global_id,
                        "image_id": image_id,
                        "x_min": round(float(x_min), 2),
                        "y_min": round(float(y_min), 2),
                        "x_max": round(float(x_max), 2),
                        "y_max": round(float(y_max), 2),
                        "confidence": round(float(top_score), 4),
                        "label": label_name,
                    }
                )

                global_id += 1

    return results


# === Save submission CSV ===
def save_submission(preds, mapping, out_csv="submission.csv"):
    df = pd.DataFrame(preds)
    df = df.drop(columns="id").drop_duplicates()

    df = df.merge(mapping[["image_id", "id"]], on="image_id", how="outer")
    df = df[
        ["id", "image_id", "x_min", "y_min", "x_max", "y_max", "confidence", "label"]
    ]
    df.to_csv(out_csv, index=False)

    print(f"✅ Saved submission to {out_csv}")


# === Main Script ===
if __name__ == "__main__":
    # Configuration
    test_dir = "test"
    model_path = "best_model.pth"
    id_mapping_path = "ID_to_Image_Mapping.csv"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classes (same order as used during training)
    classes = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    transform = create_transforms()

    # Load test mapping
    mapping_df = pd.read_csv(id_mapping_path)

    # Load test set
    test_images = mapping_df["image_id"].tolist()
    test_dataset = ChestXrayTestDataset(test_dir, test_images, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_model(model_path, device)

    # Run predictions
    predictions = run_inference(model, test_loader, classes, device, mapping_df)

    # Save CSV
    save_submission(predictions, mapping_df)
