import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(train_loader)

    return average_loss, accuracy


def evaluate(model, data_loader, device):
    model.eval()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)

    return average_loss, accuracy


def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        print(
            "epoch:",
            epoch + 1,
            "train loss:",
            round(train_loss, 4),
            "train acc:",
            round(train_accuracy, 4),
            "val loss:",
            round(val_loss, 4),
            "val acc:",
            round(val_accuracy, 4)
        )

    return model