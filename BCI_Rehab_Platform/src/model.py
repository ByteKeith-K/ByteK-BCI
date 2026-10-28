import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=22, Samples=1001, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25):
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution (learns frequencies)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Convolution (learns spatial filters/electrode locations)
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2: Separable Convolution (learns complex patterns)
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Classification Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (Samples // 32), nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x