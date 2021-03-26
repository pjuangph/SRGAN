import torch
import torch.nn.functional
from torch import flatten
from torch.nn import Sequential,Module,Linear,BatchNorm2d,Upsample,Conv2d, PixelShuffle, LeakyReLU,AdaptiveAvgPool2d
from torch.nn.modules import batchnorm 
import torchvision
import math

# https://github.com/deepak112/Keras-SRGAN/blob/7989cda237e01ab6ce3d02c3c2f9ed0cf5f4a612/Network.py#L51


class Discriminator(Module):
    def __init__(self,channels:int=64):
        super(Discriminator, self).__init__()
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

            AdaptiveAvgPool2d(1),
            Conv2d(512, 1024, kernel_size=1),   # This is similar to dense because kernel is 1 
            LeakyReLU(0.2),
            Conv2d(1024, 1, kernel_size=1)      # should output a single value for each batch 
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size)) 
        
