# FULLY CONVOLUTIONAL NETWORK FOR TIME SERIES CLASSIFICATION WITH VARIBLE LENGTH
# IMPLEMENTING THE PAPER: Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline
# AUTHOR: Andres Vicente Arevalo
# DATE: 2024-04-23

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(FCNN, self).__init__()

        # Convolutional layers of kernel sizes 8, 5, 3
        # and filter sizes 128, 256, 128
        # followed by batch normalization and ReLU activation
        # and finally global average pooling and softmax activation

        print('-' * 50)
        print('  Creating the FCNN model')
        print(f'  Input channels: {input_channels}')
        print(f'  Output classes: {output_classes}')
        print('-' * 50)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 12, output_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 12)  # Reshape for fully connected layer
        x = self.relu(self.fc1(x))
        return x.softmax(dim=1)
