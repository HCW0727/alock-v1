import torch
import torch.nn as nn

class CNN_Base(nn.Module):
    '''
    Image Classification task를 위한 가장 간단한 모델입니다.
    conv2d 기반으로 제작하였습니다.
    Cross Entropy를 사용하기에, Softmax를 사용하지 않고 Logit 그대로 반환합니다.
    '''
    def __init__(self, input_tensor):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=input_tensor,
                              out_channels=64,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.cnn2 = nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 24 * 24 ,100)

    def forward(self, input_tensor):
        # block 1
        block1_cnn = self.cnn1(input_tensor)
        block1_act = self.relu(block1_cnn)
        block1_maxpool = self.maxpool(block1_act)

        # block 2
        block2_cnn = self.cnn2(block1_maxpool)
        block2_act = self.relu(block2_cnn)
        block2_maxpool = self.maxpool(block2_act)

        # output block
        out_flattened = self.flatten(block2_maxpool)
        out_final = self.fc1(out_flattened)

        return out_final

class FCN_Base(nn.Module):
    def __init__(self, input_tensor):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_tensor,
                            out_features=128)
        
        self.fc2 = nn.Linear(in_features=128,
                            out_features=64)
        
        self.fc3 = nn.Linear(in_features=64,
                            out_features=32)
        
        self.fc4 = nn.Linear(in_features=32,
                            out_features=1)
        
        self.relu = nn.ReLU()
        
    def forward(self,input_tensor):
        # block 1
        block1_fc = self.fc1(input_tensor)
        block1_relu = self.relu(block1_fc)

        # block 2
        block2_fc = self.fc2(block1_relu)
        block2_relu = self.relu(block2_fc)

        # block 3
        block3_fc = self.fc3(block2_relu)
        block3_relu = self.relu(block3_fc)

        # block 4
        block4_fc = self.fc4(block3_relu)
        block4_relu = self.relu(block4_fc)

        return block4_fc