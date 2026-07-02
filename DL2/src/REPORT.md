# Ergebnisse: CNN Hyperparameter Optimization (Oxford-IIIT Pet)

## Dataset

Der Oxford-IIIT Pet Datensatz wurde ueber `torchvision.datasets.OxfordIIITPet`
geladen (37 Klassen, Kombination aus Katzen- und Hunderassen).

* trainval-split: ueber torchvision, 80/20 in train/val gesplittet, seed=42
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

*(optuna_results.csv nicht gefunden - bitte zuerst optuna_search.py ausfuehren)*

Beste gefundene Konfiguration:

```json
{}
```

**TODO:** kurze Diskussion - welcher Parameter hatte den groessten Einfluss?
(z.B. aus optuna_results.csv Korrelationen/Trends beschreiben, ggf. mit
`optuna.visualization.plot_param_importances(study)` einen Plot erzeugen)

## Finale Evaluation des besten Modells

Trainingseinstellungen des finalen Laufs:

```json
{}
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

*(misclassifications.csv nicht gefunden - bitte zuerst main.py ausfuehren)*

**TODO:** kurze Interpretation - welche Rassen werden am haeufigsten
verwechselt und warum (aehnliches Aussehen, schlechte Bildqualitaet, etc.)?

## Reproducibility

* Seed 42 fuer `random`, `numpy`, `torch` (inkl. CUDA), siehe `seed.py`
* `torch.backends.cudnn.deterministic = True`
* Trainingskonfiguration des finalen Modells in `final_training_settings.json`
  dokumentiert
* Modellgewichte gespeichert in `best_model.pt`
