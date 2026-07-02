# g) Systematische Hyperparameter-Optimierung mit Optuna
#
# Untersucht alle geforderten Hyperparameter:
#   - learning rate
#   - batch size
#   - dropout rate (inkl. "kein Dropout" als Option)
#   - pooling type (max / avg)
#   - Anzahl und Groesse der Conv-Layer
#
# Installation: pip install optuna
import json
import optuna
import pandas as pd
import torch

from data import get_dataloaders
from model import PetCNN
from train import train_model, evaluate
from seed import set_seed  # i)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEARCH_EPOCHS = 3  # bewusst klein gehalten, da pro Trial ein komplettes Training laeuft


def objective(trial):
    # -- Hyperparameter-Raum -----------------------------------------
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.2, 0.3, 0.5])
    pooling_type = trial.suggest_categorical("pooling_type", ["max", "avg"])

    num_layers = trial.suggest_int("num_layers", 2, 4)
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
    channels = tuple(base_channels * (2 ** i) for i in range(num_layers))

    augmentation = trial.suggest_categorical("augmentation", [True, False])
    # -------------------------------------------------------------

    set_seed(42)  # i) jeder Trial startet mit dem gleichen Seed

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        image_size=64,
        augmentation=augmentation,
    )

    model = PetCNN(
        num_classes=37,
        dropout_rate=dropout_rate,
        channels=channels,
        pooling_type=pooling_type,
    ).to(DEVICE)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=SEARCH_EPOCHS,
        learning_rate=learning_rate,
    )

    val_loss, val_accuracy = evaluate(model, val_loader, DEVICE)

    # Zwischenergebnisse in trial.user_attrs speichern, fuer die Auswertung
    trial.set_user_attr("channels", channels)
    trial.set_user_attr("val_loss", val_loss)

    return val_accuracy


def run_search(n_trials=25):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print()
    print("Bestes Trial:")
    print("  val_accuracy:", study.best_trial.value)
    print("  params:", study.best_trial.params)
    print("  channels:", study.best_trial.user_attrs.get("channels"))

    # Alle Trials als Tabelle speichern (fuer den Bericht: Tabellen/Visualisierungen)
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)
    print()
    print("Alle Trial-Ergebnisse gespeichert in optuna_results.csv")

    # i) beste Konfiguration dokumentieren
    best_config = dict(study.best_trial.params)
    best_config["channels"] = study.best_trial.user_attrs.get("channels")
    best_config["val_accuracy"] = study.best_trial.value
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print("Beste Konfiguration gespeichert in best_config.json")

    return study


if __name__ == "__main__":
    run_search(n_trials=25)
