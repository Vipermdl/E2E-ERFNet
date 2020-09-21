import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

class SE_Block(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SE_Block, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size)
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        out = torch.sigmoid(out)
        return out * x

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class UnShuffle_Layer(nn.Module):
    def __init__(self, ratios):
        super(UnShuffle_Layer, self).__init__()
        self.ratios = ratios

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        out_channels = channels * self.ratios
        out_width = in_width // self.ratios
        input_view = x.contiguous().view(batch_size, channels, in_height, self.ratios, out_width)
        shuffle_out = input_view.permute(0, 1, 3, 2, 4).contiguous()
        return shuffle_out.view(batch_size, out_channels, in_height, out_width)


class HRM_Block(nn.Module):
    def __init__(self, in_planes=96, stride=2, kernel_size=3):
        super(HRM_Block, self).__init__()
        self.horizontal_avg_pool = nn.AvgPool2d((1, stride))
        self.horizontal_convbn = BasicConv(in_planes=in_planes, out_planes=in_planes, kernel_size=1)
        self.unshuffle_layer = UnShuffle_Layer(ratios=stride)
        padding = (kernel_size - 1) // 2
        self.unshuffle_convbn = BasicConv(in_planes=in_planes*stride, out_planes=in_planes, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = SE_Block(exp_size=in_planes)
        self.drop_block = nn.Dropout(p=0.1)#(block_size=3, drop_prob=0.1)

    def forward(self, x):
        horizontal_x = self.horizontal_avg_pool(x)
        horizontal_x = self.horizontal_convbn(horizontal_x)
        unshuffle_x = self.unshuffle_layer(x)
        unshuffle_x = self.unshuffle_convbn(unshuffle_x)
        x = horizontal_x + unshuffle_x
        x = self.relu(x)
        x = self.se_block(x)
        x = self.drop_block(x)
        return x



if __name__ == '__main__':
    x = torch.randn(size=(1, 96, 128, 256))
    model = HRM_Block()
    model(x)