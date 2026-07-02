# Erzeugt REPORT.md aus den Ergebnisdateien, die von eda.py, optuna_search.py
# und main.py gespeichert wurden. Vor dem Ausfuehren muessen diese drei
# Skripte bereits gelaufen sein.
import json
import os
import pandas as pd


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"WARNUNG: {path} nicht gefunden.")
    return default


def format_optuna_table(csv_path="optuna_results.csv", top_k=10):
    if not os.path.exists(csv_path):
        return "*(optuna_results.csv nicht gefunden - bitte zuerst optuna_search.py ausfuehren)*"

    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith("params_") or c == "value"]
    df = df[cols].sort_values("value", ascending=False).head(top_k)
    df.columns = [c.replace("params_", "") for c in df.columns]
    df = df.rename(columns={"value": "val_accuracy"})
    return df.to_markdown(index=False)


def format_misclassifications(csv_path="misclassifications.csv", top_k=10):
    if not os.path.exists(csv_path):
        return "*(misclassifications.csv nicht gefunden - bitte zuerst main.py ausfuehren)*"
    df = pd.read_csv(csv_path).head(top_k)
    return df.to_markdown(index=False)


def build_report():
    best_config = load_json("best_config.json", {})
    final_settings = load_json("final_training_settings.json", {})

    report = f"""# Ergebnisse: CNN Hyperparameter Optimization (Oxford-IIIT Pet)

## Dataset

Der Oxford-IIIT Pet Datensatz wurde ueber `torchvision.datasets.OxfordIIITPet`
geladen (37 Klassen, Kombination aus Katzen- und Hunderassen).

* trainval-split: {final_settings.get('note_trainval', 'ueber torchvision, 80/20 in train/val gesplittet, seed=42')}
* test-split: offizieller `test`-split von torchvision

**TODO:** Ergebnisse aus der EDA hier kurz zusammenfassen
(siehe `eda_class_distribution.png`, `eda_image_sizes.png`, `eda_sample_images.png`):
* Klassenverteilung: ausgeglichen oder nicht?
* Bildgroessen: wie stark variierend?
* Auffaelligkeiten in Beispielbildern (Qualitaet, Hintergrund, Beleuchtung)?

## Preprocessing und Augmentation

* Resize auf feste Groesse, Normalisierung mit ImageNet-Statistiken
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
* Augmentation (nur auf Trainingsdaten): RandomHorizontalFlip

**TODO:** kurzer Vergleich augmentation=True vs. augmentation=False
(z.B. aus den Optuna-Trials mit gleichen sonstigen Parametern filtern,
oder separat mit `experiments.py` laufen lassen).

## Baseline CNN

Architektur (`model.py`, Klasse `PetCNN`):
* konfigurierbare Anzahl/Groesse an Conv-Bloecken (Conv2d + ReLU + Pooling)
* Pooling-Typ waehlbar (Max- oder Average-Pooling)
* AdaptiveAvgPool2d(4,4) vor dem Classifier
* Classifier: Linear -> ReLU -> optional Dropout -> Linear (Output: 37 Klassen)

Baseline-Konfiguration (ohne Dropout, 3 Conv-Bloecke):
channels=(32, 64, 128), pooling_type=max, dropout_rate=0.0

## Hyperparameter-Suche (Optuna)

Untersuchte Parameter: learning rate, batch size, dropout rate, pooling type,
Anzahl/Groesse der Conv-Layer, augmentation.

Top Trials nach Validation-Accuracy:

{format_optuna_table()}

Beste gefundene Konfiguration:

```json
{json.dumps(best_config, indent=2)}
```

**TODO:** kurze Diskussion - welcher Parameter hatte den groessten Einfluss?
(z.B. aus optuna_results.csv Korrelationen/Trends beschreiben, ggf. mit
`optuna.visualization.plot_param_importances(study)` einen Plot erzeugen)

## Finale Evaluation des besten Modells

Trainingseinstellungen des finalen Laufs:

```json
{json.dumps(final_settings, indent=2)}
```

**TODO:** finale Metriken aus der main.py-Konsolenausgabe hier eintragen:

| Metrik | Wert |
|---|---|
| Accuracy | ... |
| Precision (macro) | ... |
| Recall (macro) | ... |

Confusion Matrix:

![Confusion Matrix](confusion_matrix.png)

Haeufigste Verwechslungen:

{format_misclassifications()}

**TODO:** kurze Interpretation - welche Rassen werden am haeufigsten
verwechselt und warum (aehnliches Aussehen, schlechte Bildqualitaet, etc.)?

## Reproducibility

* Seed 42 fuer `random`, `numpy`, `torch` (inkl. CUDA), siehe `seed.py`
* `torch.backends.cudnn.deterministic = True`
* Trainingskonfiguration des finalen Modells in `final_training_settings.json`
  dokumentiert
* Modellgewichte gespeichert in `best_model.pt`
"""

    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("REPORT.md erstellt.")
    print("Bitte alle TODO-Abschnitte manuell ausfuellen.")


if __name__ == "__main__":
    build_report()
