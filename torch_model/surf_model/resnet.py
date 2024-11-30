import torch
from torch import nn
from torchvision import models
def Downsample(in_dim, out_dim):
    """
    下采样函数，用于在ResNet中构建残差块的下采样部分。

    参数:
    in_dim (int): 输入通道数。
    out_dim (int): 输出通道数。

    返回:
    nn.Sequential: 包含卷积和批量归一化操作的序列。
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class SurfResNet18(nn.Module):
    """
    SurfResNet18网络定义，基于ResNet18架构进行修改。

    参数:
    num_param (int): 额外输入参数的数量，用于回归层。

    Note : 输入尺寸为32的倍数,否则会丢失信息
    """

    def __init__(self, num_param):
        super(SurfResNet18, self).__init__()
        self.preprocess = nn.Sequential(
            # input : 2 * 1024*1024
            nn.Conv2d(2, 2, kernel_size=9, stride=2),  # 2 * 508 * 508
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),  # 2 * 254 * 254
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
        )

        # ResNet的四个层，每个层包含两个BasicBlock
        self.layer1 = nn.Sequential(
            models.resnet.BasicBlock(inplanes=64, planes=64, downsample=None),
            models.resnet.BasicBlock(inplanes=64, planes=64, downsample=None),
        )
        self.layer2 = nn.Sequential(
            models.resnet.BasicBlock(
                inplanes=64, planes=128, stride=2, downsample=Downsample(64, 128)
            ),
            models.resnet.BasicBlock(inplanes=128, planes=128, downsample=None),
        )
        self.layer3 = nn.Sequential(
            models.resnet.BasicBlock(
                inplanes=128, planes=256, stride=2, downsample=Downsample(128, 256)
            ),
            models.resnet.BasicBlock(inplanes=256, planes=256, downsample=None),
        )
        self.layer4 = nn.Sequential(
            models.resnet.BasicBlock(
                inplanes=256, planes=512, stride=2, downsample=Downsample(256, 512)
            ),
            models.resnet.BasicBlock(inplanes=512, planes=512, downsample=None),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.regression = nn.Linear(
            in_features=512 + num_param, out_features=2, bias=True
        )

    def forward(self, x, paras):
        """
        前向传播函数。

        参数:
        x (Tensor): 输入数据张量。
        paras (Tensor): 额外的输入参数张量。

        返回:
        Tensor: 经过网络处理后的输出张量。
        """
        x = self.preprocess(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        p = paras.view(paras.size(0), -1)
        x = torch.cat((x, p), 1)
        x = self.regression(x)
        return x
