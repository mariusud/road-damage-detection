import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *(torch.tanh(nn.functional.softplus(x)))



class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class SSD512(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS 
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.l2_norm = L2Norm(512, scale=20)
        
        backbone = wide_resnet50_2(pretrained=True)
        
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)
        
        #Additional downsampling 
        self.bank1.add_module("downsample_0",nn.Conv2d(1024,512, kernel_size=1))
        
        self.bank8 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 256,
                    out_channels = 128,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(128),
                Mish(),
                nn.Conv2d(
                    in_channels = 128,
                    out_channels = 256,
                    kernel_size=4,
                    padding=1
                ),
                nn.BatchNorm2d(256),
                Mish(),
            
                )
        self.bank2 = self._build_layer(self.output_channels[0], self.output_channels[1],2, 1024)
        self.bank3 = self._build_layer(self.output_channels[1], self.output_channels[2],3, 512)
        self.bank4 = self._build_layer(self.output_channels[2], self.output_channels[3],4, 256)
        self.bank5 = self._build_layer(self.output_channels[3],self.output_channels[4],5, 128)
        self.bank6 = self._build_layer(self.output_channels[4],self.output_channels[5],6, 128)
        #self.bank7 = self._build_layer(self.output_channels[5],self.output_channels[6],7, 128, stride=1,padding=0)
        
        self.feature_extractor = nn.ModuleList([self.bank1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6, self.bank8])
    
    def _build_layer(self, input_size,output_size, layer_no, channels, stride=2, padding=1, dilation=0):
        layer = nn.Sequential(
                nn.Conv2d(
                    in_channels = input_size,
                    out_channels = channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(channels),
                Mish(),
                nn.Conv2d(
                    in_channels = channels,
                    out_channels = output_size,
                    kernel_size=3,
                    stride=stride,
                    padding=padding
                ),
                nn.BatchNorm2d(output_size),
                Mish(),
            
                )
        return layer

    def forward(self, x):
        out = []
        for idx, feature in enumerate(self.feature_extractor):
            x = feature(x)
            if idx == 0:
                x = self.l2_norm(x)  # Conv4_3 L2 normalization
            out.append(x)

        return tuple(out)
