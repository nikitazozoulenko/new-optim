import torch
import torch.nn as nn
import numpy as np

class ResNet32x32(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super(ResNet32x32, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.n_classes=n_classes

        self.layers = nn.Sequential(self._make_residual_layer(16, scaling = "downsample"),
                                    self._make_residual_layer(16, scaling = "same"),
                                    self._make_residual_layer(16, scaling = "same"),
                                    self._make_residual_layer(32, scaling = "downsample"),
                                    self._make_residual_layer(32, scaling = "same"),
                                    self._make_residual_layer(32, scaling = "same"),
                                    self._make_residual_layer(64, scaling = "downsample"),
                                    self._make_residual_layer(64, scaling = "same"),
                                    self._make_residual_layer(64, scaling = "same"),
                                    nn.Conv2d(64*self.expansion, self.n_classes, kernel_size=4))

        
    def forward(self, x):
        x =  self.layers(x)
        return x

    
    def _make_residual_layer(self, channels, scaling = "same"):
        bottleneck = Bottleneck(self.in_channels, channels, scaling, self.expansion)
        self.in_channels = channels*self.expansion
        return bottleneck


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, scaling, expansion):
        super(Bottleneck, self).__init__()
        self.scaling = scaling # "same" or "downsample"
        self.residual = False
        if in_channels == (channels*expansion):
            self.residual = True
        
        self.convbn0 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels))
        if scaling == "downsample":
            self.convbn1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride = 2, padding=1),
                                         nn.BatchNorm2d(channels))
        else: #scaling == "same"
            self.convbn1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(channels))
        
        self.convbn2 = nn.Sequential(nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels*expansion))
        self.relu = nn.ReLU(inplace = True)
        self.avg = nn.AvgPool2d(2, stride=2, padding=0)

        
    def forward(self, x):
        res = x

        out = self.convbn0(x)
        out = self.convbn1(out)
        out = self.convbn2(out)

        if self.residual:
            if self.scaling == "same":
                out += res
            elif self.scaling == "downsample":
                out += self.avg(res)

        out = self.relu(out)
        return out


if __name__=="__main__":
    model = ResNet32x32()

    x = torch.ones(1,3,32,32)
    y = model(x)
    print(y.size())
