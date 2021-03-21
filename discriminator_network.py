import torch
import torch.nn.functional
from torch import flatten
from torch.nn import Sequential,Module,Linear,BatchNorm2d,Upsample,Conv2d, PixelShuffle, LeakyReLU,AdaptiveAvgPool2d
from torch.nn.modules import batchnorm 
import torchvision
import math

# https://github.com/deepak112/Keras-SRGAN/blob/7989cda237e01ab6ce3d02c3c2f9ed0cf5f4a612/Network.py#L51

# https://github.com/leftthomas/SRGAN/blob/master/model.py 
class DiscriminatorBlock(Module):
    def __init__(self,in_channels:int=64,out_channels:int=64, kernel_size:int=3, stride:int=1):
        super(DiscriminatorBlock, self).__init__()
        self.net = Sequential(
            Conv2d(in_channels,out_channels,
                kernel_size=kernel_size,stride=stride,padding=1),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.net(x)

class Discriminator(Module):
    def __init__(self,channels:int=64):
        super(Discriminator, self).__init__()
        # self.net = Sequential(
        #     Conv2d(3, channels, kernel_size=3, padding=1),
        #     LeakyReLU(0.2),

        #     DiscriminatorBlock(channels,channels,3,2),      # 64 x 64
        #     DiscriminatorBlock(channels,2*channels,3,2),    # 64 x 128
        #     DiscriminatorBlock(2*channels,2*channels,3,2),  # 128 x 128
        #     DiscriminatorBlock(2*channels,4*channels,3,2),  # 128 x 256
        #     DiscriminatorBlock(4*channels,4*channels,3,2),  # 256 x 256
        #     DiscriminatorBlock(4*channels,8*channels,3,2),  # 256 x 512
        #     DiscriminatorBlock(8*channels,8*channels,3,2),  # 512 x 512
        #     AdaptiveAvgPool2d(1),                           # Flatten 
        #     Conv2d(8*channels, 16*channels, kernel_size=1), # Dense 1024
        #     LeakyReLU(0.2),
        #     Conv2d(16*channels, 1, kernel_size=1)           # Dense 1024 1 vector 
        # )
        self.net = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            LeakyReLU(0.2),

            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2),

            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2),
        )
        
        self.net2 = Sequential(
            Linear(512*3*3,1024),
            LeakyReLU(0.2),
            Linear(1024,1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x)
        # x1 = self.adaptive_avg_pool2D(x)
        x2 = x1.view(x1.size(0),-1)
        x3 = self.net2(x2)
        x4 = torch.sigmoid(x3)
        return x4 
        # return torch.sigmoid(self.net(x).view(batch_size))
