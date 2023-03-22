import torch.nn as nn
from model.base_net_lib import Bneck, SEblock

class MobileNetV1_Base(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_dw(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias=False)
        self.conv2_1 = conv_dw( 32, 64)
        self.conv2_2 = conv_dw( 64, 128, stride=2)
        self.conv3_1 = conv_dw(128, 128)
        self.conv3_2 = conv_dw(128, 256, stride=2)
        self.conv4_1 = conv_dw(256, 256)
        self.conv4_2 = conv_dw(256, 512)  # conv4_2
        self.conv5_1 = conv_dw(512, 512, dilation=2, padding=2)
        self.conv5_2 = conv_dw(512, 512)
        self.conv5_3 = conv_dw(512, 512)
        self.conv5_4 = conv_dw(512, 512)
        self.conv5_5 = conv_dw(512, 512)   # conv5_5 28*28*512
        self.relu = nn.ReLU()
    
    def forward(self, h):
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3(h))
        h = self.relu(self.conv5_4(h))
        h = self.relu(self.conv5_5(h))
        return h

class MobileNetV3_Base(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )

        # 中间层, 使用bneck块
        self.layer2_12 = nn.Sequential(
            Bneck(input_size=16, out_size=16, operator_kernel=3, exp_size=16, NL='RE', s=2, SE=True),
            Bneck(input_size=16, out_size=24, operator_kernel=3, exp_size=72, NL='RE', s=2, SE=False),
            Bneck(input_size=24, out_size=24, operator_kernel=3, exp_size=88, NL='RE', s=1, SE=False),
            Bneck(input_size=24, out_size=40, operator_kernel=5, exp_size=96, NL='HS', s=2, SE=True),
            Bneck(input_size=40, out_size=40, operator_kernel=5, exp_size=240, NL='HS', s=1, SE=True),
            Bneck(input_size=40, out_size=40, operator_kernel=5, exp_size=240, NL='HS', s=1, SE=True),
            Bneck(input_size=40, out_size=48, operator_kernel=5, exp_size=120, NL='HS', s=1, SE=True),
            Bneck(input_size=48, out_size=48, operator_kernel=5, exp_size=144, NL='HS', s=1, SE=True),
            Bneck(input_size=48, out_size=96, operator_kernel=5, exp_size=288, NL='HS', s=2, SE=True),
            Bneck(input_size=96, out_size=96, operator_kernel=5, exp_size=576, NL='HS', s=1, SE=True),
            Bneck(input_size=96, out_size=96, operator_kernel=5, exp_size=576, NL='HS', s=1, SE=True),
        )

        # 结尾层
        self.layer13 = nn.Sequential(
            nn.Conv2d(96, 512, 1, stride=1),
            SEblock(512),
            nn.BatchNorm2d(512),
            nn.Hardswish(),
        )

    def forward(self, h):
        h = self.layer1(h)
        h = self.layer2_12(h)
        h = self.layer13(h)
        return h

class VGG19_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_4 = nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, h):
        h = self.relu(self.conv1_1(h))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        return h