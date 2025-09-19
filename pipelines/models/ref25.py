import torch.nn as nn

class CollisionLocalizationNet(nn.Module):
    """
    A feed-forward neural network for collision localization.
    - Input: Joint torques (or other features)
    - Hidden Layer 1: 28 neurons
    - Hidden Layer 2: 14 neurons
    - Output: 7 classes (representing the link with the collision)
    """
    def __init__(self, input_size=7, num_classes=7):
        super(CollisionLocalizationNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 28)
        self.layer2 = nn.Linear(28, 14)
        self.output_layer = nn.Linear(14, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # No softmax needed here as CrossEntropyLoss will apply it internally
        x = self.output_layer(x)
        return x