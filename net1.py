import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class ECA_Layer(nn.Module):
    def __init__(self, channels):
        super(ECA_Layer, self).__init__()
        
        kernel = math.ceil((math.log(channels,2)/2 + 0.5))
        if kernel % 2 == 0:
            kernel -= 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x*y.expand_as(x)

class DSC2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class my_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        inter_channel = out_ch//4
        
        self.conv1x1 = nn.Conv2d(in_ch, inter_channel, 1, stride, 0, bias=False)
        self.conv3x3 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        self.conv5x5 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        self.conv7x7 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        
        self.channel_attention = ECA_Layer(out_ch)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(out1)
        out5 = self.conv5x5(out3)
        out7 = self.conv7x7(out5)
        
        cat = torch.cat((out1,out3,out5,out7), dim=1)
        cat = self.channel_attention(cat)
        
        return cat


class NormPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
    
    def forward(self, x):
        x = F.pad(x, self.padding, mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,))
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)# + 1e-6
        
        return mean - std


class PoolConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias):
        super(PoolConv2d, self).__init__()
        
        #self.pool = NormPool2d(kernel_size, stride, padding)
        #self.position = None
        #self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=bias)
        self.conv = my_Conv2d(in_ch, out_ch)
        
    def forward(self, x):
        #x = self.pool(x)
        #if self.position is None:
        #    num_range = x.shape[2] * x.shape[3]
        #    self.position = (torch.arange(num_range).reshape(1, 1, x.shape[2], x.shape[3]).float().cuda() / num_range) - 0.5
        
        #x = x + self.position
        x = self.conv(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2,3,32,32)).to(device)
    b = PoolConv2d(3,16,3,1,1,False).to(device)
    c = b(a)
    print(c.shape)

    
    
    