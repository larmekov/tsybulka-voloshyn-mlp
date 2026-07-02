# Finales Training und Evaluation des besten Modells (aus experiments.py)
# Aufruf: python src/evaluate_final.py
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
from seed import set_seed

RESULTS_DIR = "reports"
FINAL_EPOCHS = 15


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


def plot_confusion_matrix(matrix, class_names, path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, xticklabels=class_names, yticklabels=class_names, cmap="viridis")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsaechlich")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def analyze_misclassifications(matrix, class_names, path, top_k=10):
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
    df.to_csv(path, index=False)
    print()
    print(f"Top {top_k} haeufigste Verwechslungen:")
    print(df.head(top_k).to_string(index=False))
    return df


def load_best_config(path=f"{RESULTS_DIR}/best_config.json"):
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
        config["channels"] = tuple(config["channels"])
        print("Beste Konfiguration aus", path, "geladen:", config)
        return config
    print(f"{path} nicht gefunden (bitte zuerst experiments.py ausfuehren). "
          f"Verwende Default-Konfiguration.")
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "dropout_rate": 0.0,
        "pooling_type": "max",
        "channels": (32, 64, 128),
        "augmentation": True,
    }


def write_report(metrics, config, training_settings, misclass_df, optuna_csv):
    optuna_table = "*(nicht vorhanden)*"
    if os.path.exists(optuna_csv):
        df = pd.read_csv(optuna_csv)
        cols = [c for c in df.columns if c.startswith("params_") or c == "value"]
        df = df[cols].sort_values("value", ascending=False).head(10)
        df.columns = [c.replace("params_", "") for c in df.columns]
        df = df.rename(columns={"value": "val_accuracy"})
        optuna_table = df.to_markdown(index=False)

    misclass_table = misclass_df.head(10).to_markdown(index=False)

    report = f"""# Ergebnisse: CNN Hyperparameter Optimization (Oxford-IIIT Pet)

## Dataset

Oxford-IIIT Pet Datensatz, 37 Klassen (Katzen- und Hunderassen), geladen ueber
`torchvision.datasets.OxfordIIITPet`. trainval-Split 80/20 in train/val
(seed=42), offizieller test-Split unveraendert.

**TODO:** kurze Zusammenfassung der EDA-Ergebnisse hier einfuegen
(siehe eda_class_distribution.png, eda_image_sizes.png, eda_sample_images.png).

## Preprocessing und Augmentation

Resize + Normalisierung (ImageNet mean/std). Augmentation (nur Training):
RandomHorizontalFlip.

## Hyperparameter-Suche (Optuna)

Top Trials nach Validation-Accuracy:

{optuna_table}

Beste gefundene Konfiguration:

```json
{json.dumps(config, indent=2)}
```

**TODO:** kurze Diskussion, welcher Parameter den groessten Einfluss hatte.

## Finale Evaluation

Trainingseinstellungen:

```json
{json.dumps(training_settings, indent=2)}
```

| Metrik | Wert |
|---|---|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision (macro) | {metrics['precision']:.4f} |
| Recall (macro) | {metrics['recall']:.4f} |

Confusion Matrix:

![Confusion Matrix](confusion_matrix.png)

Haeufigste Verwechslungen:

{misclass_table}

**TODO:** kurze Interpretation der haeufigsten Verwechslungen.

## Reproducibility

Seed 42 fuer random/numpy/torch (inkl. CUDA), cudnn.deterministic=True
(siehe seed.py). Alle Trainingseinstellungen des finalen Laufs sind oben
dokumentiert und zusaetzlich in final_training_settings.json gespeichert.
Modellgewichte: best_model.pt
"""

    with open(f"{RESULTS_DIR}/results.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{RESULTS_DIR}/results.md erstellt. Bitte TODO-Abschnitte manuell ausfuellen.")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    config = load_best_config()

    training_settings = {
        "epochs": FINAL_EPOCHS,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "dropout_rate": config["dropout_rate"],
        "pooling_type": config["pooling_type"],
        "channels": list(config["channels"]),
        "augmentation": config["augmentation"],
        "seed": 42,
    }
    with open(f"{RESULTS_DIR}/final_training_settings.json", "w") as f:
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

    y_true, y_pred = get_predictions(model, test_loader, device)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    matrix = confusion_matrix(y_true, y_pred)

    print()
    print("Final evaluation")
    print("================")
    print("accuracy:", round(metrics["accuracy"], 4))
    print("precision:", round(metrics["precision"], 4))
    print("recall:", round(metrics["recall"], 4))

    class_names = get_raw_dataset(split="test").classes
    plot_confusion_matrix(matrix, class_names, f"{RESULTS_DIR}/confusion_matrix.png")
    misclass_df = analyze_misclassifications(
        matrix, class_names, f"{RESULTS_DIR}/misclassifications.csv"
    )

    torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model.pt")
    print(f"\nModell gespeichert in {RESULTS_DIR}/best_model.pt")

    write_report(
        metrics, config, training_settings, misclass_df,
        optuna_csv=f"{RESULTS_DIR}/optuna_results.csv"
    )


if __name__ == "__main__":
    main()
