import torch
from torch import nn
import torch.nn.functional as F
from sub_layers import ConvBnRelu

class SimpleSleepPPGModel(nn.Module):
    def __init__(self, num_classes=2):  # num_classes는 분류할 클래스 수
        super(SimpleSleepPPGModel, self).__init__()
        
        # Convolutional Layers
        conv_list = [
            ConvBnRelu(1, 8, 15, 5, is_max_pool=False),
            ConvBnRelu(8, 8, 15, 5, is_max_pool=False),

            ConvBnRelu(8, 8, 10, 3, is_max_pool=False), 
            ConvBnRelu(8, 8, 10, 3, is_max_pool=False), 

            ConvBnRelu(8, 8, 5, 2, is_max_pool=False),
            ConvBnRelu(8, 8, 5, 2, is_max_pool=False),

            ConvBnRelu(8, 8, 2, 2, is_max_pool=False),
            ConvBnRelu(8, 8, 2, 2, is_max_pool=False),
            ConvBnRelu(8, 8, 2, 2, is_max_pool=False),
            ConvBnRelu(8, 8, 2, 2, is_max_pool=False),
        ]
        self.conv_list = nn.ModuleList(conv_list)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 229, 32) 
        self.fc2 = nn.Linear(32, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        for _conv in self.conv_list:
            x = _conv(x)

        # Flattening
        x = torch.flatten(x, start_dim=1)
        
        # # Fully Connected Layers
        x = F.relu(self.fc1(x))  # Fully Connected Layer + ReLU
        x = self.dropout(x)      # Dropout
        x = self.fc2(x)          # Output Layer (분류)
        
        return x
    

class SimpleSleepPPGModel_2(nn.Module):
    def __init__(self, num_classes=2):  # num_classes는 분류할 클래스 수
        super(SimpleSleepPPGModel_2, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization
        self.bn8 = nn.BatchNorm1d(8)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn64 = nn.BatchNorm1d(64)
        
        # MaxPooling Layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 2, 128)  # 조정된 입력 크기 (64 feature maps, 길이 16으로 감소)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, mask):
        # Convolutional Layers with ReLU, BatchNorm, and MaxPooling
        x, mask = self.apply_layer(x, mask, self.conv1, self.bn8)
        x, mask = self.apply_layer(x, mask, self.conv2, self.bn8)
        x, mask = self.apply_layer(x, mask, self.conv3, self.bn8)
        x, mask = self.apply_layer(x, mask, self.conv4, self.bn8)
        
        x, mask = self.apply_layer(x, mask, self.conv5, self.bn16)
        x, mask = self.apply_layer(x, mask, self.conv6, self.bn32)
        x, mask = self.apply_layer(x, mask, self.conv7, self.bn64)
        
        # Flattening
        x = torch.flatten(x * mask, start_dim=1)  # Mask 적용 후 Flatten
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))  # Fully Connected Layer + ReLU
        x = self.dropout(x)      # Dropout
        x = self.fc2(x)          # Output Layer (분류)
        
        return x

    def apply_layer(self, x, mask, conv, bn):
        """
        Convolutional Layer + ReLU + BatchNorm + MaxPooling + Mask Update
        """
        x = conv(x)
        x = F.relu(bn(x))  # ReLU + BatchNorm
        x = self.pool(x)   # MaxPooling
        # mask = mask[:, :, ::2]  # MaxPooling에 따라 mask 업데이트 (stride=2)

        # MaxPooling에 따른 mask 업데이트
        input_length = mask.shape[-1]
        output_length = (input_length - self.pool.kernel_size) // self.pool.stride + 1
        mask = mask[:, :, :output_length * self.pool.stride:self.pool.stride]  # stride에 맞춰 자르기
        return x, mask


# -*- coding = utf-8 -*
# @Time：  10:07
# @File: MAF_CNN.py
# @Software: PyCharm
import torch
import torch.nn as nn

# MAF-CNN === Start =========================== 
class SELayer(nn.Module):
    def __init__(self, channel=32, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        print("SE input (x) by concat: ([32, 3], [32, 2], [32, 3]): ", x.shape)
        b, c, _ = x.size()
        print("b, c: {}, {}".format(b, c))
        y = self.avg_pool(x).view(b, c)
        print("after squeeze (squeeze the last dimension):", y.shape)
        y = self.fc(y).view(b, c, 1)
        print("after exication (y): ", y.shape)
        print("y.expand_as(x): ", y.expand_as(x).shape)
        return x * y.expand_as(x)

class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 32, (5,), (1,), dilation=(2,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, (4,), (1,), dilation=(3,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, (3,), (1,), dilation=(4,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.se = SELayer()

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x)
        # print(x2.shape)
        x3 = self.conv3(x)
        # print(x3.shape)
        out = torch.cat([x1, x2, x3], dim=2)
        # print(out.shape)
        out = self.se(out)
        # print(out.shape)
        return out

class MAF_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MAF_CNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, (256,), (32,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),

            nn.Dropout(),

            nn.Conv1d(64, 128, (7,), (1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, (7,), (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
        )

        self.dropout = nn.Dropout()
        self.msa = MSA()
        self.ft = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.out = nn.Linear(128, num_classes)

    def forward(self, x1):
        x1 = self.cnn1(x1)
        print("after CNN: ", x1.shape)

        x1 = self.msa(x1)
        print("after MSA (x * y.expand_as(x)): ", x1.shape)

        x_concat = x1
        # print(x_concat.shape)
        x = self.dropout(x_concat)
        x = self.ft(x)

        out = self.fc(x)
        x = self.out(out)
        return out, x
# MAF-CNN === End =========================== 