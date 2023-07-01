import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .quantization import *

class Base_CNN(nn.Module):
    def __init__(self, n_bit, in_features=1, num_classes=10):
        super().__init__()
        self.n_bit = n_bit
        self.activation = activation_quantize_fn(self.n_bit)

        self.conv1= Conv2d_Q(in_features, 128,
                               kernel_size=3,
                               padding=0,
                               stride=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2= Conv2d_Q(128,
                               64,
                               kernel_size=3,
                               padding=0,
                               stride=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = Conv2d_Q(64,
                        64,
                        kernel_size=3,
                        padding =0, 
                        stride =1,
                         bias =True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 =  Conv2d_Q(64, 32, kernel_size=3, padding = 0, stride = 1, bias = True)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 =  Conv2d_Q(32, 32, kernel_size=3, padding = 0, stride = 1, bias = True)
        self.bn5 = nn.BatchNorm2d(32)
        self.fc1 =  Linear_Q(32, 2000)
        #self.bn6 = nn.BatchNorm1d(2000)
        self.fc2 =  Linear_Q(2000, 100)
        #self.bn7 = nn.BatchNorm1d(100)
        self.fc3 =  Linear_Q(100, 10)

        for name, layer in self.named_modules():
            if isinstance(layer, Conv2d_Q) or isinstance(layer, Linear_Q):
                layer.set_quantization_level(self.n_bit)
    

    def forward(self, x):
        x = self.bn1(self.activation(self.conv1(x)))
        x = F.max_pool2d(x, (3, 3), 1)
        x = self.bn2(self.activation(self.conv2(x)))
        x = F.max_pool2d(x, (3, 3), 2)
        x = self.bn3(self.activation(self.conv3(x)))
        x = self.bn4(self.activation(self.conv4(x)))
        x = self.bn5(self.activation(self.conv5(x)))
        x = F.max_pool2d(x, (3, 3), 2)
        x = torch.flatten(x,1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

class CNN_simple(nn.Module):
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


