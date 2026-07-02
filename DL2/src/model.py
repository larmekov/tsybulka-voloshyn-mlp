# e) Baseline CNN
# g) erweitert um: pooling_type (max/avg) und variable Anzahl an Conv-Bloecken
import torch.nn as nn


class PetCNN(nn.Module):
    def __init__(self, num_classes=37, dropout_rate=0.0, channels=(32, 64, 128), pooling_type="max"):
        """
        channels: Tuple/Liste beliebiger Laenge -> bestimmt sowohl die Anzahl
                   der Conv-Bloecke (len(channels)) als auch deren Groesse.
        pooling_type: "max" oder "avg"
        """
        super().__init__()

        if pooling_type == "max":
            pool_layer = nn.MaxPool2d
        elif pooling_type == "avg":
            pool_layer = nn.AvgPool2d
        else:
            raise ValueError("pooling_type muss 'max' oder 'avg' sein")

        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(pool_layer(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        last_channels = channels[-1]
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(last_channels * 4 * 4, 256),
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
