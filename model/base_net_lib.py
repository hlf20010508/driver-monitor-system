import torch
import torch.nn as nn

class Bneck(nn.Module):
    def __init__(self, input_size, out_size, operator_kernel, exp_size, NL, s, SE=False):
        """
        MobileNetV3的block块
        :param input_size: 输入维度
        :param operator_kernel: Dw卷积核大小
        :param exp_size: 升维维数
        :param out_size: 输出维数
        :param NL: 非线性激活函数,包含Relu以及h-switch
        :param s: 卷积核步矩
        :param SE: 是否使用注意力机制,默认为false
        :param skip_connection: 是否进行跳跃连接,当且仅当输入与输出维数相同且大小相同时开启
        """
        super().__init__()
        # 1.使用1×1卷积升维
        self.conv_1_1_up = nn.Conv2d(input_size, exp_size, 1)
        if NL == 'RE':
            self.nl1 = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(exp_size),
            )
        elif NL == 'HS':
            self.nl1 = nn.Sequential(
                nn.Hardswish(),
                nn.BatchNorm2d(exp_size),
            )
        # 2.使用Dwise卷积, groups与输入输出维度相同
        self.depth_conv = nn.Conv2d(exp_size, exp_size, kernel_size=operator_kernel, stride=s, groups=exp_size,
                                    padding=(operator_kernel - 1) // 2)  # 进行padding补零使shape减半时保存一致
        self.nl2 = nn.Sequential(
            self.nl1,
            nn.BatchNorm2d(exp_size)
        )

        #  3.使用1×1卷积降维
        self.conv_1_1_down = nn.Conv2d(exp_size, out_size, 1)

        # 判断是否添加注意力机制
        self.se = SE
        if SE:
            self.se_block = SEblock(exp_size)

        # 判断是否使用跳跃连接: 说明-> 当输入维数不等于输出维数且大小相同时才进行跳跃连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size),
        )
        if input_size != out_size and s == 1:
            self.skip = True
        else:
            self.skip = False

    def forward(self, x):
        # 1.1×1卷积升维
        x1 = self.conv_1_1_up(x)
        x1 = self.nl1(x1)

        # 2.Dwise卷积
        x2 = self.depth_conv(x1)
        x2 = self.nl2(x2)

        # 3.1×1卷积降维
        x3 = self.conv_1_1_down(x2)

        # 判断是否添加注意力机制
        if self.se:
            x2 = self.se_block(x2)

        # 判断是否使用跳跃连接
        if self.skip:
            x3 = x3 + self.shortcut(x)
        return x3


class SEblock(nn.Module):
    def __init__(self, channel, r=0.25):
        """
        注意力机制模块
        :param channel: channel为输入的维度,
        :param r: r为全连接层缩放比例->控制中间层个数 默认为1/4
        """
        super().__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),  # 原文中的hard-alpha 我不知道是什么激活函数,就用SE原文的Sigmoid替代(如果你知道是什么就把这儿的激活函数替换掉)
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        scale = weight * x
        return scale
