import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, input_tensor):
        super().__init__()

        # Conv2d Layers
        self.conv1_1 = nn.Conv2d(in_channels=input_tensor, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # Linear
        self.fc6 = nn.Linear(in_features=4608, out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=10)

        # Utils
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()



    def forward(self, input_tensor):
        b1 = self.relu(self.conv1_1(input_tensor))
        b1 = self.relu(self.conv1_2(b1))
        b1 = self.maxpool(b1)

        b2 = self.relu(self.conv2_1(b1))
        b2 = self.relu(self.conv2_2(b2))
        b2 = self.maxpool(b2)

        b3 = self.relu(self.conv3_1(b2))
        b3 = self.relu(self.conv3_2(b3))
        b3 = self.relu(self.conv3_3(b3))
        b3 = self.maxpool(b3)

        b4 = self.relu(self.conv4_1(b3))
        b4 = self.relu(self.conv4_2(b4))
        b4 = self.relu(self.conv4_3(b4))
        b4 = self.maxpool(b4)

        b5 = self.relu(self.conv5_1(b4))
        b5 = self.relu(self.conv5_2(b5))
        b5 = self.relu(self.conv5_3(b5))
        b5 = self.maxpool(b5)

        b6 = self.flatten(b5)
        b6 = self.relu(self.fc6(b6))
        b7 = self.relu(self.fc7(b6))
        b8 = self.fc8(b7)

        return b8