# from train import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import feature_alpha_dropout

class FeatureExtractionModel(nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractionModel,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return x



class GazePredictModel(nn.Module):
    def __init__(self):
        super(GazePredictModel,self).__init__()
        self.features = FeatureExtractionModel()
        self.fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,3),
        )

    def forward(self,x):
        x = self.features(x)
        # x.view(x.size(0),-1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = GazePredictModel()
    print(model)