import torch
import torch.nn as nn
import torchvision.models.video as models_video

class ResNet3DClassifier(nn.Module):
    def __init__(self, config):
        super(ResNet3DClassifier, self).__init__()

        # 사전학습된 r3d_18 불러오기
        self.model = models_video.r3d_18(weights=models_video.R3D_18_Weights.KINETICS400_V1)

        # Stem 부분 수정 (Grayscale 입력 처리 - in_channels=1)
        self.model.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        # FC 레이어 수정
        self.model.fc = nn.Linear(512, config.num_classes)

    def forward(self, x):
        # Stem 부터 Layer4까지 feature 추출
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Average Pooling
        x = self.model.avgpool(x)
        embedding = x.flatten(1)  # FC 입력 전 feature vector (임베딩)

        out = self.model.fc(embedding)  # Class Prediction

        return out, embedding