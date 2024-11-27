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

