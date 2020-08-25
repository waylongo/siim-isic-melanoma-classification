import torch
import torch.nn.functional as F
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Net(nn.Module):
    def __init__(self, depth=0):
        super(Net, self).__init__()
        self.base = EfficientNet.from_pretrained(f"efficientnet-b{depth}")
        self.in_features = self.base._fc.in_features
        self.base._fc = nn.Linear(self.in_features, 384, bias=False)

        self.meta_layer = nn.Sequential(
            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.hidden = nn.Sequential(
            nn.Linear(384 + 32, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.ouput = nn.Linear(384, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = F.relu(self.base(x))
        meta_features = self.meta_layer(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        features = F.relu(self.hidden(features))
        output = self.ouput(features)

        return output
