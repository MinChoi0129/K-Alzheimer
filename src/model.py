import torch
import torch.nn as nn
import torchvision.models.video as models_video


class ResNet3DClassifier(nn.Module):
    def __init__(self, config):
        super(ResNet3DClassifier, self).__init__()

        self.model = models_video.r3d_18(weights=models_video.R3D_18_Weights.KINETICS400_V1)
        self.model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model.avgpool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Dropout(p=0.5))
        self.model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        embedding = x.flatten(1)
        out = self.model.fc(embedding)

        return out, embedding
