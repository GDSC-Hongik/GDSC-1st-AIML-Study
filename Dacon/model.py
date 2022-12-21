import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.linear1 = nn.Linear(1000, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        out = self.model(x)
        out = self.linear1(out)
        out = self.linear2(out)

        return out


class AttentionBasedMIL(nn.Module):
    def __init__(self):
        super(AttentionBasedMIL, self).__init__()
        self.L = 512  # 512 node fully connected layer
        self.D = 128  # 128 node attention layer
        self.K = 1
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * 30 * 30, self.L),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 48 * 30 * 30)
        H = self.feature_extractor_part2(H)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)
        # The probability that a given bag is malignant or benign
        Y_prob = self.classifier(M)
        # The prediction given the probability (Y_prob >= 0.5 returns a Y_hat of 1 meaning malignant)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A.byte()

# FIXME : Will be Changed to InceptionV3
class MyInceptionV3(nn.Module):
    """Concat된 애들을 AdativePooling 229x229로 만들어 준 다음에 inception v3에 넣습니다. """
    """일단 임시로 efnb0으로 했습니다 ㅠㅠ inceptionv3 뭔가 좀 다르네요...?"""
    def __init__(self):
        super(MyInceptionV3, self).__init__()

        self.pooling = nn.AdaptiveAvgPool2d((299, 299))
        self.main = models.efficientnet_b0(pretrained=True)
        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 256)

    def forward(self, x) :
        x = self.pooling(x)
        x = self.main(x) # Inception V3's Output : x, Aux -> 나중에 x, _ 로!
        x = self.linear1(x)
        output = self.linear2(x)
        return output

class TabularExtractor(nn.Module):
    def __init__(self):
        super(TabularExtractor, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_features=23, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256)
        )
        
    def forward(self, x):
        output = self.embedding(x)
        return output

class ModalClassifier(nn.Module):
    def __init__(self):
        super(ModalClassifier, self).__init__()

        self.img_team = MyInceptionV3()
        self.tab_team = TabularExtractor()
        self.finalize = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256,2)
        )

    def forward(self, img, tabular):
        img_feature = self.img_team(img)
        tab_feature = self.tab_team(tabular)
        flatten_feature = torch.cat([img_feature, tab_feature], dim=-1)
        output = self.finalize(flatten_feature)
        return output