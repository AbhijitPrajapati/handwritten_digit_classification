from torch import nn


layers = nn.Sequential(
            # (bsize, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same'),
            # (bsize, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # (bsize, 32, 14, 14)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            # (bsize, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # (bsize, 64, 7, 7)
            nn.Flatten(),
            # (bsize, 64*7*7)
            nn.Linear(in_features=64*7*7, out_features=128),
            # (bsize, 128)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=10)
            # (bsize, 10)
        )


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = layers
    
    def forward(self, x): return self.layers(x)