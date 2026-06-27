import torch

from data import get_dataloaders
from model import PetCNN
from train import train_model, evaluate


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        image_size=64,
        augmentation=False
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
        epochs=3,
        learning_rate=0.001
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    print("test loss:", round(test_loss, 4))
    print("test accuracy:", round(test_accuracy, 4))


if __name__ == "__main__":
    main()