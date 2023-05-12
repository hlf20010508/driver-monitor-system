import torch
import torch.nn as nn
from torchvision import models
# import model.base_net as base_net

class DMNet(nn.Module):
    def __init__(self, heatmap_num, paf_num):
        super().__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1(heatmap_num, paf_num)
        self.stage_2 = Stage_2(heatmap_num, paf_num)
        
    def forward(self, h):
        feature_map = self.base(h)
        h1, h2 = self.stage_1(feature_map)
        h1, h2 = self.stage_2(torch.cat([h1, h2, feature_map], dim = 1))
        heatmaps = torch.squeeze(h1)
        pafs = torch.squeeze(h2)
        return heatmaps, pafs

class Cpm(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
                nn.ELU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.ELU(inplace=True),
            )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(128, 128),
            conv_dw_no_bn(128, 128),
            conv_dw_no_bn(128, 128)
        )
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, h):
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h + self.trunk(h)))
        return h

class Base_model(nn.Module):
    def __init__(self):
        super().__init__()
        # shape: (height/16) X (width/16)
        self.net_base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features[:9]
        self.cpm = Cpm(48)
    def forward(self, h):
        h = self.net_base(h)
        h = self.cpm(h)
        return h
    
class Stage_1(nn.Module):
    def __init__(self, heatmap_num, paf_num):
        super().__init__()
        self.conv1_CPM = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # heatmap
        self.conv4_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = nn.Conv2d(in_channels=512, out_channels=heatmap_num, kernel_size=1, stride=1, padding=0) # 28*28*heatmap_num
        # paf
        self.conv4_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L2 = nn.Conv2d(in_channels=512, out_channels=paf_num, kernel_size=1, stride=1, padding=0) # 28*28*paf_num
        self.relu = nn.ReLU()
        
    def forward(self, h):
        h = self.relu(self.conv1_CPM(h))
        h = self.relu(self.conv2_CPM(h))
        h = self.relu(self.conv3_CPM(h))
        h1 = self.relu(self.conv4_CPM_L1(h)) # heatmap
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv4_CPM_L2(h)) # paf
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2

class Stage_2_trunk(nn.Module):
    def __init__(self, in_channels = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size=1, stride = 1, padding=0)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride = 1, padding=2, dilation=2, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(128)
    def forward(self, h):
        h1 = self.relu(self.conv1(h))
        h2 = self.relu(self.bn(self.conv2(h1)))
        h2 = self.relu(self.bn(self.conv3(h2)))
        return h1 + h2

class Stage_2(nn.Module):
    def __init__(self, heatmap_num, paf_num):
        super().__init__()
        self.conv1 = Stage_2_trunk(heatmap_num + paf_num + 128)
        self.conv2 = Stage_2_trunk()
        self.conv3 = Stage_2_trunk()
        self.conv4 = Stage_2_trunk()
        self.conv5 = Stage_2_trunk() # 28*28*128
        # heatmap
        self.conv6_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L1 = nn.Conv2d(in_channels = 128, out_channels = heatmap_num, kernel_size = 1, stride = 1, padding = 0) # 28*28*heatmap_num
        # paf
        self.conv6_L2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L2 = nn.Conv2d(in_channels = 128, out_channels = paf_num, kernel_size = 1, stride = 1, padding = 0) # 28*28*paf_num
        self.relu = nn.ReLU()
        
    def forward(self, h):
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        h1 = self.relu(self.conv6_L1(h)) # heatmap
        h1 = self.conv7_L1(h)
        h2 = self.relu(self.conv6_L2(h)) # paf
        h2 = self.conv7_L2(h)
        return h1, h2