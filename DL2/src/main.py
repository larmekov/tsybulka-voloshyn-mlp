# h) Finale Evaluation des besten Modells, i) Reproducibility
import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from data import get_dataloaders, get_raw_dataset
from model import PetCNN
from train import train_model
from seed import set_seed  # i)


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


def plot_confusion_matrix(matrix, class_names):
    # h) Visualisierung der Confusion Matrix als Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, xticklabels=class_names, yticklabels=class_names, cmap="viridis")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsaechlich")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def analyze_misclassifications(matrix, class_names, top_k=10):
    # h) Analyse der haeufigsten Verwechslungen (ohne die Diagonale, also echte Fehler)
    rows = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                rows.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": matrix[i, j],
                })
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    print()
    print(f"Top {top_k} haeufigste Verwechslungen:")
    print(df.head(top_k).to_string(index=False))
    df.to_csv("misclassifications.csv", index=False)
    return df


def load_best_config(path="best_config.json"):
    # g)/i) Konfiguration, die von optuna_search.py gefunden wurde
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
        config["channels"] = tuple(config["channels"])
        print("Beste Konfiguration aus", path, "geladen:", config)
        return config
    print(f"{path} nicht gefunden, verwende Default-Konfiguration.")
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "dropout_rate": 0.0,
        "pooling_type": "max",
        "channels": (32, 64, 128),
        "augmentation": True,
    }


def main():
    set_seed(42)  # i)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    config = load_best_config()

    # i) verwendete Einstellungen dokumentieren
    training_settings = {
        "epochs": 15,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "dropout_rate": config["dropout_rate"],
        "pooling_type": config["pooling_type"],
        "channels": list(config["channels"]),
        "augmentation": config["augmentation"],
        "seed": 42,
    }
    with open("final_training_settings.json", "w") as f:
        json.dump(training_settings, f, indent=2)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        image_size=64,
        augmentation=config["augmentation"],
    )

    model = PetCNN(
        num_classes=37,
        dropout_rate=config["dropout_rate"],
        channels=config["channels"],
        pooling_type=config["pooling_type"],
    ).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=training_settings["epochs"],
        learning_rate=config["learning_rate"],
    )

    y_true, y_pred = get_predictions(model=model, data_loader=test_loader, device=device)

    # h) Metriken
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

    class_names = get_raw_dataset(split="test").classes
    plot_confusion_matrix(matrix, class_names)
    analyze_misclassifications(matrix, class_names)

    torch.save(model.state_dict(), "best_model.pt")
    print()
    print("Modell gespeichert in best_model.pt")


if __name__ == "__main__":
    main()
