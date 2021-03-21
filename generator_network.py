''' 
    Photo-Realistic Single Image Super-REsolution 
    Using Generative Adversarial Networks 
'''
import torch
import torch.nn.functional
from torch import flatten
from torch.nn import Sequential,Module,PReLU,BatchNorm2d,Upsample,Conv2d, PixelShuffle, LeakyReLU,AdaptiveAvgPool2d 
import torchvision
import math


class Generator(Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = Sequential(
            Conv2d(3, 64, kernel_size=9, padding=4),
            PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = Sequential(
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        y = (torch.tanh(block8) + 1) / 2
        return y

'''Residual block 

'''
class ResidualBlock(Module):
    def __init__(self, channels=64):
        '''Residual Blocks 
            These are linked together. Uses 3x3 Kernels, 64 feature maps followed by batch normalization step 
            
            Args
                Channels - (default 64) these are feature maps
        '''
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(channels)
        self.prelu = PReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual # Element wise summation 


class UpsampleBLock(Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle(up_scale)
        self.prelu = PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

