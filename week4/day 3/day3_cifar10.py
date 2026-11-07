"""
Week 4 - Day 3: CIFAR-10 final CNN training
==========================================

Trains a CNN with data augmentation, BatchNorm, Dropout and cosine LR decay.
Saves:
- cifar_cnn.pt                 (model state_dict)
- cifar_cnn_checkpoint.pth     (full resumable checkpoint)
- cifar_training_curves.png
- cifar_confusion_matrix.png
- cifar_test_report.txt

Run:
    python week4/day3_cifar10.py

Recommended: Python 3.11/3.12 with matching torch + torchvision versions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


WEEK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = WEEK_DIR / "results" / "day3"
DATA_DIR = WEEK_DIR / "data" / "cifar10"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.05
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CIFAR10CNN(nn.Module):
    """Compact CNN with BatchNorm and Dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_loaders():
    # CIFAR-10 channel statistics.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform,
    )

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_and_confusion_matrix(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)

    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)

    accuracy = float((y_true == y_pred).mean())
    return accuracy, matrix


def save_training_curves(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("CIFAR-10 CNN - Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cifar_training_loss.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
    plt.plot(epochs, history["test_accuracy"], label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 CNN - Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cifar_training_accuracy.png", dpi=250)
    plt.close()


def save_confusion_matrix(matrix):
    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=45, ha="right")
    plt.yticks(ticks, CLASSES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = matrix.max() * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cifar_confusion_matrix.png", dpi=250)
    plt.close()


def main():
    set_seed()
    print("=" * 68)
    print("WEEK 4 - DAY 3: CIFAR-10 CNN TRAINING")
    print("=" * 68)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS}")

    train_loader, test_loader = make_loaders()
    model = CIFAR10CNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "learning_rate": [],
    }

    best_accuracy = 0.0
    best_state = None
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc*100:.2f}%"
        )

        # Save resumable checkpoint after every epoch.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_test_accuracy": best_accuracy,
            "history": history,
            "classes": CLASSES,
        }
        torch.save(
            checkpoint,
            RESULTS_DIR / "cifar_cnn_checkpoint.pth"
        )

    training_time = time.perf_counter() - start_time

    # Restore best-performing model before exporting/evaluating.
    if best_state is not None:
        model.load_state_dict(best_state)

    state_path = RESULTS_DIR / "cifar_cnn.pt"
    torch.save(model.state_dict(), state_path)

    # Also save a full checkpoint corresponding to the selected best weights.
    final_checkpoint = {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_test_accuracy": best_accuracy,
        "history": history,
        "classes": CLASSES,
    }
    torch.save(
        final_checkpoint,
        RESULTS_DIR / "cifar_cnn_checkpoint.pth"
    )

    save_training_curves(history)

    # Explicitly reload state_dict into a fresh model and test it.
    reloaded_model = CIFAR10CNN().to(DEVICE)
    try:
        loaded_state = torch.load(
            state_path, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        loaded_state = torch.load(state_path, map_location=DEVICE)
    reloaded_model.load_state_dict(loaded_state)

    final_accuracy, confusion = evaluate_and_confusion_matrix(
        reloaded_model, test_loader
    )
    save_confusion_matrix(confusion)

    report = (
        "CIFAR-10 FINAL EVALUATION\n"
        "========================\n"
        f"Device: {DEVICE}\n"
        f"Epochs: {EPOCHS}\n"
        f"Best validation/test accuracy during training: {best_accuracy*100:.2f}%\n"
        f"Reloaded model test accuracy: {final_accuracy*100:.2f}%\n"
        f"Training time (seconds): {training_time:.2f}\n"
        f"Target >= 70%: {'PASSED' if final_accuracy >= 0.70 else 'NOT YET REACHED'}\n"
    )
    (RESULTS_DIR / "cifar_test_report.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n" + report)
    print(f"Saved model: {state_path}")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
