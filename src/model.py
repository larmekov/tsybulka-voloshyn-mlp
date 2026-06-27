import torch.nn as nn


class PetCNN(nn.Module):
    def __init__(self, num_classes=37, dropout_rate=0.0, channels=(32, 64, 128)):
        super().__init__()

        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            nn.ReLU(),
        ]

        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))

        classifier_layers.append(nn.Linear(256, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x