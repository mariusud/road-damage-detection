import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# https://github.com/bicycleman15/SSD-ResNet-PyTorch/blob/master/models/resnet50_backbone.py
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py


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

class ResNet_Experimental(nn.Module):
    # The resnet backbone is to be used as a feature provider.
    # For 300x300 we expect a 38x38 feature map

    def __init__(self, cfg):
        super().__init__()
        # Loading the full Resnet 50 backbone. This means that the we are currently only
        # importing the REsnet50s architecture. No weights or biases have been initialised.
        # WE will be using Xavier initialisation for that
        # Loading the full Resnet 50 backbone
        #backbone = resnet50(pretrained=True)
        self.out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS #[1024, 512, 512, 256, 256, 256] # resnet50
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.l2_norm = L2Norm(1024, scale=20)

        '''
        # Extracting the the required layers form the backbone
        # nn.sequential converts the individial components extracted from resnet in a list to
        # to continiuos nn object on which we can perform backprop
        # Lets call this as our feautre provider. This provied us with the very first feature map [38x38] with 1024 channels
        self.feature_provider = nn.Sequential(*list(backbone.children())[:7])
        
        # NOTE: But it is necessary to change the layer's stride else the feature provider will give a feature of 19x19
        # The conv4_x layer is the last object in our feature provider list.
        # Since stride arvariable in only the first block of a resnet layer we select the
        # last layer with the [-1] index and its first block with [0] in the self.feature_provider[-1][0]
        conv4_block1 = self.feature_provider[-1][0]

        conv4_block1.conv1.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.conv2.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.downsample[0].stride = (
            1, 1)  # changing the stride to 1x1
        
        '''
        
        backbone = resnet50(pretrained=True)
        
        # out of bank1 -> 1024 x 38 x 38
        # source https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])
        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)

        # HELT BASIC EXTRA FEATURE LAYERS
        # +BATCHNORM and switched ReLU order
        # out of bank2 -> 512 x 19 x 19
        '''
        # Custom convolution layers
        self.bank2 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[1]),
        )
        # out -> 512 x 10 x 10
        self.bank3 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[2]),
        )
        # out -> 256 x 5 x 5
        self.bank4 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[2],
                out_channels = self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[3]),
        )
        # out of bank5 -> 256 x 3 x 3
        self.bank5 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[3],
                out_channels = self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[4]),
        )
        # out of bank6 -> 128 x 1 x 1
        self.bank6 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[4],
                out_channels = self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[5]),
        )'''
        
        self.bank2 = self._build_layer(self.output_channels[0], self.output_channels[1],2, 256)
        self.bank3 = self._build_layer(self.output_channels[1], self.output_channels[2],3, 256)
        self.bank4 = self._build_layer(self.output_channels[2], self.output_channels[3],4, 128)
        self.bank5 = self._build_layer(self.output_channels[3],self.output_channels[4],5, 128)
        self.bank6 = self._build_layer(self.output_channels[4],self.output_channels[5],6, 128, stride=1,padding=0)
        self.feature_extractor = nn.ModuleList([self.bank1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6])
    
    def _build_layer(self, input_size,output_size, layer_no, channels, stride=2, padding=1):
        '''self.additional_blocks = []
        if layer_no <= 3:
            layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            
        else:
            layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
        '''
        
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels = input_size,
                out_channels = output_size,
                kernel_size=3,
                stride=stride,
                padding=padding
            ),
            Mish(),
            nn.BatchNorm2d(output_size),
        )
        return layer

    def forward(self, x):
        out_features = []
        idx=0
        for feature in self.feature_extractor:

            x = feature(x)
            if idx == 0:
                x = self.l2_norm(x)  # Conv4_3 L2 normalization

            out_features.append(x)
            idx +=1

        """
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)
