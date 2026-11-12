
"""
Week 4 — Day 1
CIFAR-10 CNN with Train / Validation / Test Split

Dataset:
- 45,000 training images
- 5,000 validation images
- 10,000 test images

Features:
- CNN with BatchNorm and Dropout
- Data augmentation
- Train and validation accuracy
- Loss and accuracy graphs
- Best model checkpoint
- Final test evaluation
"""

from pathlib import Path
import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "cifar10"

RESULTS_DIR = ROOT / "results" / "cifar10"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"

for folder in [PLOTS_DIR, MODELS_DIR, METRICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.001

RANDOM_SEED = 42

NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. TRANSFORMS
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    return train_transform, test_transform


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_data():

    print("\n[1] Loading CIFAR-10...")

    train_transform, test_transform = get_transforms()

    train_dataset_augmented = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform
    )

    train_dataset_clean = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=test_transform
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform
    )

    # Reproducible split
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        len(train_dataset_augmented),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(
        train_dataset_augmented,
        train_indices
    )

    val_dataset = Subset(
        train_dataset_clean,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ============================================================
# 4. CNN ARCHITECTURE
# ============================================================

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 7. TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    model_path = MODELS_DIR / "best_cifar10_cnn.pth"

    print("\n[2] Starting training...")
    print(f"Device: {DEVICE}")

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "classes": CLASSES
                },
                model_path
            )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc * 100:.2f}% "
            f"| Best Val: {best_val_accuracy * 100:.2f}%"
        )

    print("\nBest validation accuracy:",
          f"{best_val_accuracy * 100:.2f}%")

    return history, model_path


# ============================================================
# 8. TEST EVALUATION
# ============================================================

def test_model(model, model_path, test_loader):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n[3] Final test evaluation")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": checkpoint["val_accuracy"],
        "best_epoch": checkpoint["epoch"]
    }


# ============================================================
# 9. TRAINING GRAPHS
# ============================================================

def save_plots(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "loss.png", dpi=200)
    plt.close()

    # Accuracy
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        np.array(history["train_accuracy"]) * 100,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        np.array(history["val_accuracy"]) * 100,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "accuracy.png", dpi=200)
    plt.close()

    print("Training graphs saved.")


# ============================================================
# 10. SAVE METRICS
# ============================================================

def save_metrics(history, test_results):

    csv_path = METRICS_DIR / "training_history.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])

        for i in range(len(history["train_loss"])):

            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_accuracy"][i],
                history["val_loss"][i],
                history["val_accuracy"][i]
            ])

    with open(
        METRICS_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(test_results, file, indent=4)

    print("CNN metrics saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 55)
    print("WEEK 4 — DAY 1: CIFAR-10 CNN")
    print("=" * 55)

    train_loader, val_loader, test_loader = load_data()

    model = CIFAR10CNN().to(DEVICE)

    history, model_path = train_model(
        model,
        train_loader,
        val_loader
    )

    test_results = test_model(
        model,
        model_path,
        test_loader
    )

    save_plots(history)

    save_metrics(history, test_results)

    print("\n" + "=" * 55)
    print("CIFAR-10 TRAINING COMPLETED")
    print("=" * 55)

    if test_results["test_accuracy"] >= 0.70:
        print("Target achieved: Test accuracy >= 70%")
    else:
        print("Target not reached yet. Consider further training.")

    print(
        f"Final test accuracy: "
        f"{test_results['test_accuracy'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
