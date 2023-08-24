import numpy
import torch
import torch.nn as nn

from mixstyle import MixStyle
from common import l2_norm, norm_feature, RandomReplace, ChannelShuffleCustom, RandomZero


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features


class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        mix="crosssample",
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a, mix=mix)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3", "layer4"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
            print(f'Using {mix}')
        else:
            print('No MixStyle')
        self.ms_layers = ms_layers

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x, labels)

        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x, labels)

        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x, labels)

        x = self.layer4(x)
        if "layer4" in self.ms_layers:
            x = self.mixstyle(x, labels)

        return x

    def forward(self, x, labels=None):
        f = self.featuremaps(x, labels)
        return f


def init_pretrained_weights(model, model_url):
    print(model_url)
    #pretrain_dict = model_zoo.load_url(model_url, progress=True)
    pretrain_dict = torch.hub.load_state_dict_from_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=2):
        super(Classifier, self).__init__()

        self.classifier_layer = nn.Linear(in_channels, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=False):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            output = self.classifier_layer(input)
        else:
            output = self.classifier_layer(input)
        return output


class MixStyleResCausalModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False, num_classes=2, prob=0.2, ms_class=MixStyle, ms_layers=["layer1", "layer2"], mix="crosssample"):
        super(MixStyleResCausalModel, self).__init__()
        self.feature_extractor = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            ms_class=ms_class,
            ms_layers=ms_layers, #["layer1", "layer2"], # "layer3", "layer4"
            mix=mix
        )
        if pretrained:
            print('-------load model----------')
            init_pretrained_weights(self.feature_extractor, model_urls[model_name])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = Classifier(in_channels=512, num_classes=num_classes)   # resnet

        self.RandomReplace = RandomReplace(p=prob)
        self.dropout = RandomZero(p=prob)
        self.channel_shuffle = ChannelShuffleCustom(groups=16)

    def forward(self, input, labels=None, cf=['cs', 'dropout', 'replace'], norm=True):

        feature = self.feature_extractor(input, labels)  # Batch, 512, 7, 7

        cls_feature = self.avgpool(feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls = self.classifier(cls_feature)

        if (not self.training) or cf is None or labels is None:
            return cls
        else:
            used_m = numpy.random.choice(cf) #, 'mix', 'cs', 'dropout', 'replace'
            if used_m == 'dropout':
                cf_output = self.dropout(feature.clone()) #.clone()  'clone with no norm works'
            elif used_m == 'cs':
                cf_output = self.channel_shuffle(feature) #.clone() # no clone
            elif used_m == 'replace':
                cf_output = self.RandomReplace(feature.clone()) #.clone() # need clone
            else:
                raise Exception('Error: No counterfactural operation given.')
            cf_output = self.avgpool(cf_output)
            if norm:
                cf_output = norm_feature(cf_output, p=2, dim=1)
            cf_output = cf_output.view(cf_output.size(0), -1)
            cls_cf_out = self.classifier(cf_output)
            return cls, cls_cf_out

def _test():
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #image_x = torch.randn(16, 3, 224, 224) # 112-9, 224-16, 112-112-16
    labels = torch.tensor([0, 1, 1, 0])
    image_x = torch.randn(1, 3, 224, 224)
    #listmodels = timm.list_models('conv*', pretrained=True)
    #print(listmodels)
    model = MixStyleResModel(ms_layers=[])
    model = resnet18(num_classes=2)
    flops, params = profile(model, inputs=(image_x, ))

    print(flops, params)
    #print('FLIPs (G):', flops_to_string(flops, units='GFLOPS', precision=4))
    #print('Param (M):', params / 1e6)

    exit()


    #model = BaseModel('convnext_small')
    #model.train()
    #output = model(image_x)
    #print(output.shape)

if __name__ == "__main__":
    _test()
