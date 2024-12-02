import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from os import path
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict


class AdaptorNet1024(nn.Module):
    """
    AdaptorNet类, 将输入参数处理为需要的形状

    对于现代的神经网络, 其输入尺寸应当为32的倍数
    """

    def __init__(self):
        # 初始化AdaptorNet类的构造函数
        super(AdaptorNet1024, self).__init__()

        # 初始化卷积层序列
        self.conv = nn.Conv2d(2, 2, kernel_size=7, padding=5, stride=2)
        # input : 2 * 1024*1024
        # 使用2个输入通道，2个输出通道，kernel_size=9的卷积核，stride=2的步长进行卷积操作
        # (1024 - 7 + 2 * 5) / 2 + 1 = 514
        # 输出尺寸变为2 * 514 * 514

        # 应用ReLU激活函数，对输入进行非线性变换，加速神经网络的训练
        self.relu = nn.ReLU(inplace=True)
        # 应用最大池化操作，kernel_size=3的卷积核，stride=2的步长，不使用向上取整模式
        # (514 - 3) / 2 + 1 = 256
        # 输出尺寸变为2 * 256 * 256
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class ModifiedPretrainedNet(nn.Module):
    """
    修改的预训练网络类。

    该类用于根据指定的预训练网络结构，修改其第一个卷积层和全连接层，以适应特定的任务需求。

    参数:
        - pretrained_net (nn.Module): 预训练的网络模型。
        - name_first_conv (str): 需要修改的第一个卷积层的名称。
        - name_fc (str): 需要修改的全连接层的名称。
        - weights (Optional[Dict]): 预训练模型的权重，如果有的话。
    """

    def __init__(
        self,
        pretrained_net: nn.Module,
        name_first_conv: str,
        name_fc: str,
        weights: Optional[Dict] = None,
    ):
        super(ModifiedPretrainedNet, self).__init__()
        # 初始化预训练网络，并根据提供的权重进行加载（如果有）
        self.pretrained_net = pretrained_net(weights=weights)
        conv_module = None
        fc_module = None
        # 遍历预训练网络的所有模块，找到需要修改的卷积层和全连接层
        for name, module in self.pretrained_net.named_modules():
            if isinstance(module, nn.Conv2d) and name == name_first_conv:
                conv_module = module

            elif isinstance(module, nn.Linear) and name == name_fc:
                fc_module = module

        # 获取卷积层的属性
        out_channels = conv_module.out_channels
        kernel_size = conv_module.kernel_size
        stride = conv_module.stride
        padding = conv_module.padding
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # 替换预训练网络中的第一个卷积层
        parent_module = self.pretrained_net
        parts = name_first_conv.split(".")
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, parts[-1], new_conv)

        # 获取全连接层的输入特征数，并替换全连接层为Identity，为后续任务自定义回归层
        self.in_features = fc_module.in_features
        parent_module = self.pretrained_net
        parts = name_fc.split(".")
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name.split(".")[-1], nn.Identity())

    def forward(self, x):
        return self.pretrained_net(x)


class FeatureParamsCombinedRegression(nn.Module):
    """
    特征和参数组合的回归模型。

    该模型旨在通过融合特征和参数的处理，对数据进行非线性变换，并最终进行回归预测。

    参数:
    - feature_size: int, 输入特征的维度。
    - num_params: int, 输入参数的数量。
    - num_output: int, 输出的维度。
    - dropout: float = 0.5, dropout概率, 默认为0.5。
    - cnn_feature_ratio: float = 0.5, cnn feature向量和参数feature向量之间的长度比例, 默认为0.5。
    """

    def __init__(
        self,
        feature_size: int,
        num_params: int,
        num_output: int,
        dropout: float = 0.5,
        cnn_feature_ratio: float = 0.5,
    ):
        # 初始化父类
        super(FeatureParamsCombinedRegression, self).__init__()
        # 计算隐藏层大小，这里选择将特征尺寸减半
        cnn_feature_size = round(feature_size * cnn_feature_ratio)
        param_size = feature_size - cnn_feature_size
        hide_size = feature_size // 2
        # 特征压缩层, 将表面特征数量压缩为一半
        self.feature_reduction = nn.Sequential(
            nn.Linear(feature_size, cnn_feature_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(cnn_feature_size),
        )

        # 参数扩展层, 将输入参数扩展为hidden_size
        self.param_expansion = nn.Sequential(
            nn.Linear(num_params, param_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(param_size),
        )

        # 特征合并层, 将特征和参数合并 (是否需要?)
        self.combine_layer = nn.Sequential(
            nn.Linear(feature_size, hide_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hide_size),
            nn.Dropout(dropout),
        )

        # 回归层
        self.regression = nn.Linear(hide_size, num_output)

    def forward(self, features, params):
        f = torch.flatten(features, 1)
        p = params.view(params.size(0), -1)
        # 将预训练网络的输出与额外参数合并，并通过回归层得到最终输出
        f = self.feature_reduction(f)
        p = self.param_expansion(p)
        x = torch.cat((f, p), dim=1)
        x = self.combine_layer(x)
        x = self.regression(x)
        return x


class SurfNet1024(nn.Module):
    """
    SurfNet1024类是一个用于处理特定任务的神经网络模型，结合了预训练网络和自定义模块以实现端到端的学习和预测。

    输入尺寸为1024*1024*2, 因此需要使用一个adaptor模块将输入尺寸调整到256*256*2。

    参数:
    - modified_net (ModifiedPretainedNet): 一个经过修改的预训练网络实例，作为此模型的一部分。
    - num_params (int): 输入到模型的参数数量，用于预测输出。
    - num_output (int): 模型的输出维度。
    - dropout (float): 在输出层应用的dropout概率，默认值为0.5。
    - cnn_feature_ratio: float = 0.5, cnn feature向量和参数feature向量之间的长度比例, 默认为0.5。
    """

    def __init__(
        self,
        modified_net: ModifiedPretrainedNet,
        num_params: int,
        num_output: int,
        dropout: float = 0.5,
        cnn_feature_ratio: float = 0.5,
    ):
        # 初始化父类
        super(SurfNet1024, self).__init__()
        # 定义适配器模块，用于处理输入数据
        self.adaptor = AdaptorNet1024()
        # 初始化预训练网络
        self.pretrained_net = modified_net
        # 初始化回归层和模块引用
        self.output = FeatureParamsCombinedRegression(
            self.pretrained_net.in_features, num_params, num_output, dropout
        )

    def forward(self, x, params):
        """
        前向传播函数，处理输入数据并返回模型的预测结果。

        参数:
        - x: 输入到模型的数据。
        - params: 额外的输入参数,与x一起用于预测。

        返回:
        - x: 模型的预测结果。
        """
        # 对输入数据进行预处理和前向传播
        x = self.adaptor(x)
        x = self.pretrained_net(x)
        x = self.output(x, params)
        return x


class SurfNet256(nn.Module):
    """
    SurfNet256类是一个用于处理特定任务的神经网络模型，结合了预训练网络和自定义模块以实现端到端的学习和预测。

    输入尺寸为256*256*2, 因此不需要使用adaptor模块。

    参数:
    - modified_net (ModifiedPretainedNet): 一个经过修改的预训练网络实例，作为此模型的一部分。
    - num_params (int): 输入到模型的参数数量，用于预测输出。
    - num_output (int): 模型的输出维度。
    - dropout (float): 在输出层应用的dropout概率，默认值为0.2。
    - cnn_feature_ratio: float = 0.5, cnn feature向量和参数feature向量之间的长度比例, 默认为0.5。
    """

    def __init__(
        self,
        modified_net: ModifiedPretrainedNet,
        num_params: int,
        num_output: int,
        dropout: float = 0.2,
        cnn_feature_ratio: float = 0.5,
    ):
        # 初始化父类
        super(SurfNet256, self).__init__()
        # 初始化预训练网络
        self.pretrained_net = modified_net
        # 初始化回归层和模块引用
        self.output = FeatureParamsCombinedRegression(
            self.pretrained_net.in_features, num_params, num_output, dropout
        )

    def forward(self, x, params):
        """
        前向传播函数，处理输入数据并返回模型的预测结果。

        参数:
        - x: 输入到模型的数据。
        - params: 额外的输入参数,与x一起用于预测。

        返回:
        - x: 模型的预测结果。
        """
        # 对输入数据进行预处理和前向传播
        x = self.pretrained_net(x)
        x = self.output(x, params)
        return x
