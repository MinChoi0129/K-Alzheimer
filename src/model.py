import torch
import torch.nn as nn
import torchvision.models.video as models_video


class ResNet3DClassifier(nn.Module):
    def __init__(self, config):
        super(ResNet3DClassifier, self).__init__()

        self.backbone = models_video.r3d_18(weights=models_video.R3D_18_Weights.KINETICS400_V1)
        self.backbone.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        self.backbone.avgpool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Dropout(p=0.5))
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.num_classes)

        visual_feat_dim = self.backbone.fc.in_features
        demo_embed_dim = 32
        self.demographic_fc = nn.Sequential(nn.Linear(2, demo_embed_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.1))
        self.classifier = nn.Linear(visual_feat_dim + demo_embed_dim, config.num_classes)

    def forward(self, x, sex, age):  # [BS, 1, 32, 224, 224]
        x = self.backbone.stem(x)  # [BS, 64, 32, 112, 112]
        x = self.backbone.layer1(x)  # [BS, 64, 32, 112, 112]
        x = self.backbone.layer2(x)  # [BS, 128, 16, 56, 56]
        x = self.backbone.layer3(x)  # [BS, 256, 8, 28, 28]
        x = self.backbone.layer4(x)  # [BS, 512, 4, 14, 14]
        x = self.backbone.avgpool(x)  # [BS, 512, 1, 1, 1]
        visual_embedding = x.flatten(1)  # [BS, 512]

        sex_in = sex.view(-1, 1).float()
        age_in = age.view(-1, 1).float()
        demo_input = torch.cat([sex_in, age_in], dim=1)
        demo_emb = self.demographic_fc(demo_input)

        combined_emb = torch.cat([visual_embedding, demo_emb], dim=1)
        class_logits = self.classifier(combined_emb)

        return class_logits, combined_emb
