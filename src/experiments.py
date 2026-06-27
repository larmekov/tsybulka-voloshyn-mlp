import torch

from data import get_dataloaders
from model import PetCNN
from train import train_model, evaluate


def run_experiment(name, augmentation, dropout_rate, learning_rate, channels, epochs=3):
    print()
    print("Experiment:", name)
    print("augmentation:", augmentation)
    print("dropout_rate:", dropout_rate)
    print("learning_rate:", learning_rate)
    print("channels:", channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        image_size=64,
        augmentation=augmentation
    )

    model = PetCNN(
        num_classes=37,
        dropout_rate=dropout_rate,
        channels=channels
    )

    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate
    )

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        device=device
    )

    result = {
        "name": name,
        "augmentation": augmentation,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "channels": channels,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    print("test loss:", round(test_loss, 4))
    print("test accuracy:", round(test_accuracy, 4))

    return result


def main():
    torch.manual_seed(42)

    experiments = [
        {
            "name": "baseline",
            "augmentation": False,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "channels": (32, 64, 128),
        },
        {
            "name": "augmentation",
            "augmentation": True,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "channels": (32, 64, 128),
        },
        {
            "name": "dropout_0_3",
            "augmentation": True,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "channels": (32, 64, 128),
        },
        {
            "name": "lower_lr",
            "augmentation": True,
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "channels": (32, 64, 128),
        },
        {
            "name": "bigger_cnn",
            "augmentation": True,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "channels": (64, 128, 256),
        },
    ]

    results = []

    for experiment in experiments:
        result = run_experiment(
            name=experiment["name"],
            augmentation=experiment["augmentation"],
            dropout_rate=experiment["dropout_rate"],
            learning_rate=experiment["learning_rate"],
            channels=experiment["channels"],
            epochs=3
        )

        results.append(result)

    print()
    print("Summary")
    print("=======")

    for result in results:
        print(
            result["name"],
            "test accuracy:",
            round(result["test_accuracy"], 4),
            "test loss:",
            round(result["test_loss"], 4)
        )


if __name__ == "__main__":
    main()