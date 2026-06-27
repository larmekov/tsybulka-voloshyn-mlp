import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from data import get_dataloaders
from model import PetCNN
from train import train_model


def get_predictions(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        image_size=64,
        augmentation=True
    )

    model = PetCNN(
        num_classes=37,
        dropout_rate=0.0,
        channels=(32, 64, 128)
    )

    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5,
        learning_rate=0.001
    )

    y_true, y_pred = get_predictions(
        model=model,
        data_loader=test_loader,
        device=device
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    print()
    print("Final evaluation")
    print("================")
    print("accuracy:", round(accuracy, 4))
    print("precision:", round(precision, 4))
    print("recall:", round(recall, 4))

    print()
    print("confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()