import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
)

# ----------------------------- REPRODUCIBILITY -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------- CONFIGURATION -----------------------------
DATASET_ROOT = Path("./dataset")     # <-- CHANGE THIS TO YOUR DATASET FOLDER
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
MIXUP_ALPHA = 0.2
PATIENCE = 5
NUM_WORKERS = 0                      # keep 0 on Windows to avoid multiprocessing errors


# =========================== 1. PARSE DATASET ==============================
def parse_dataset(root):
    records = []
    for split_folder in root.iterdir():
        if not split_folder.is_dir():
            continue
        for label_name in ["Fracture", "Normal"]:
            folder = split_folder / label_name
            if not folder.exists():
                continue
            label_int = 1 if label_name == "Fracture" else 0
            for img_path in folder.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                records.append({
                    "path": str(img_path),
                    "label": label_int,
                    "split_original": split_folder.name,
                    "filename": img_path.name
                })
    return pd.DataFrame(records)

df = parse_dataset(DATASET_ROOT)
print(f"Total images found: {len(df)}")
print(df["label"].value_counts())


# ========================= 2. PATIENT-AWARE SPLIT ==========================
def extract_patient_id(filename, label):
    stem = Path(filename).stem
    if label == 1:  # fracture: "1 Male (A View).jpg"
        parts = stem.split()
        return str(parts[0])
    else:           # normal: "1.jpg"
        try:
            return f"N_{int(stem)}"
        except ValueError:
            return f"N_{stem}"

df["patient_id"] = df.apply(
    lambda r: extract_patient_id(r["filename"], r["label"]), axis=1
)
df["patient_id"] = df["patient_id"].astype(str)

print("Unique patients:", df["patient_id"].nunique())

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")
print("Train class distribution:", train_df["label"].value_counts().to_dict())
print("Val   class distribution:", val_df["label"].value_counts().to_dict())


# ========================= 3. AUGMENTATIONS ================================
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
    A.GaussNoise(std_range=(0.001, 0.01), p=0.3),
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(0.02, 0.07),   # 4/224 ≈ 0.02, 16/224 ≈ 0.07
        hole_width_range=(0.02, 0.07),
        fill_value=0,
        p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# =========================== 4. DATASET & LOADERS ==========================
class WristDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, row["label"]

train_dataset = WristDataset(train_df, transform=train_transforms)
val_dataset = WristDataset(val_df, transform=val_transforms)

# Weighted sampling
train_labels = train_df["label"].values
class_counts = np.bincount(train_labels)
print("Train class counts:", class_counts)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)


# =========================== 5. MODEL ======================================
def get_last_conv_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model.")
    return [last_conv]

model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=1)
model = model.to(device)
target_layers = get_last_conv_layer(model)

pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


# =========================== 6. MIXUP ======================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)    # <-- FIX: randperm not randperm
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========================= 7. TRAINING FUNCTIONS ===========================
def train_one_epoch(loader, model, criterion, optimizer, scheduler, mixup_alpha):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    scheduler.step()
    return running_loss / len(loader.dataset)

def validate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    return running_loss / len(loader.dataset), all_preds, all_labels

def compute_metrics(preds, labels, threshold=0.5):
    pred_binary = (preds >= threshold).astype(int)
    acc = accuracy_score(labels, pred_binary)
    prec = precision_score(labels, pred_binary, zero_division=0)
    rec = recall_score(labels, pred_binary, zero_division=0)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    auc = roc_auc_score(labels, preds)
    return acc, prec, rec, f1, auc


# ========================= 8. PREDICTION FUNCTION ==========================
def predict_single_image(image_path, model, transform, device, target_layers):
    """Predict fracture on a single image and return label, confidence and heatmap."""
    model.eval()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to match model input size for Grad-CAM overlay
    image_resized = cv2.resize(image_rgb, (224, 224))
    orig_image = image_resized / 255.0

    # Transform for model
    augmented = transform(image=image_resized)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    label = "Fracture" if prob >= 0.5 else "Normal"
    confidence = prob if label == "Fracture" else 1 - prob

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    heatmap = show_cam_on_image(orig_image, grayscale_cam, use_rgb=True)

    return label, confidence, heatmap


# ========================= 9. MAIN PIPELINE ================================
def main():
    # ---------- Training ----------
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    print("\n====== START TRAINING ======")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer,
                                     scheduler, MIXUP_ALPHA)
        val_loss, val_preds, val_labels = validate(val_loader, model, criterion)
        acc, prec, rec, f1, auc = compute_metrics(val_preds, val_labels)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            print("  🔥 Best model updated")
        else:
            early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("⛔ Early stopping triggered!")
            break

    model.load_state_dict(best_model_wts)
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")

    # ---------- Final Evaluation ----------
    model.eval()
    val_preds_final, val_labels_final = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            val_preds_final.extend(preds)
            val_labels_final.extend(labels.numpy())

    val_preds_final = np.array(val_preds_final).flatten()
    val_labels_final = np.array(val_labels_final).flatten()
    acc, prec, rec, f1, auc = compute_metrics(val_preds_final, val_labels_final)

    print("\n====== FINAL VALIDATION METRICS ======")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(val_labels_final, (val_preds_final >= 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fracture'], yticklabels=['Normal', 'Fracture'])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(val_labels_final, val_preds_final)
    plt.figure()
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # ---------- Grad-CAM on validation samples ----------
    cam = GradCAM(model=model, target_layers=target_layers)

    def load_image_for_gradcam(img_path):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 for consistent overlay
        image_resized = cv2.resize(image, (224, 224))
        orig_image = image_resized / 255.0
        augmented = val_transforms(image=image_resized)   # transform already includes resize
        input_tensor = augmented["image"].unsqueeze(0).to(device)
        return input_tensor, orig_image

    def show_gradcam(img_paths, title_prefix="Fracture"):
        fig, axes = plt.subplots(1, len(img_paths), figsize=(15, 5))
        if len(img_paths) == 1:
            axes = [axes]
        for ax, path in zip(axes, img_paths):
            input_tensor, orig_img = load_image_for_gradcam(path)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            heatmap = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
            ax.imshow(heatmap)
            ax.axis('off')
            ax.set_title(f"{title_prefix} - {Path(path).name}")
        plt.tight_layout()
        plt.show()

    # Show Grad-CAM for a few fracture and normal examples (if available)
    fracture_examples = val_df[val_df["label"] == 1]
    normal_examples = val_df[val_df["label"] == 0]
    if len(fracture_examples) > 0:
        print("\nGrad-CAM for Fracture samples:")
        show_gradcam(fracture_examples.sample(min(3, len(fracture_examples)), random_state=42)["path"].values, "Fracture")
    if len(normal_examples) > 0:
        print("Grad-CAM for Normal samples:")
        show_gradcam(normal_examples.sample(min(3, len(normal_examples)), random_state=42)["path"].values, "Normal")

    # ---------- Save final model and artifacts ----------
    artifact = {
        "model_state_dict": model.state_dict(),
        "val_transform": val_transforms.to_dict(),
        "class_names": ["Normal", "Fracture"]
    }
    torch.save(artifact, "deployment_artifacts.pth")
    print("\n✅ Model and artifacts saved to 'deployment_artifacts.pth'")

    # ---------- INTERACTIVE PREDICTION LOOP ----------
    print("\n==============================================")
    print("   🔍 CUSTOM IMAGE PREDICTION ")
    print("   Enter the path to a wrist X-ray image")
    print("   Type 'quit' or 'exit' to stop.")
    print("==============================================")

    while True:
        img_path = input("\nImage path: ").strip().strip('"')
        if img_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not os.path.isfile(img_path):
            print("❌ File does not exist. Please try again.")
            continue

        try:
            label, confidence, heatmap = predict_single_image(
                img_path, model, val_transforms, device, target_layers
            )
            print(f"Result: {label} (confidence: {confidence:.2%})")

            # Display the heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap)
            plt.title(f"Prediction: {label} ({confidence:.2%})")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Pipeline finished.")


if __name__ == "__main__":
    main()