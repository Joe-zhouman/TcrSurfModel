import torch
from torch import nn


class SurfAlexNet(nn.Module):
    def __init__(self, num_params):
        """
        初始化SurfAlexNet类的实例。

        参数:
        - num_params: int，用于回归头部的额外输入特征数量。
        """
        super(SurfAlexNet, self).__init__()
        # 定义特征提取部分，使用Sequential容器组织多个卷积和池化层
        self.features = nn.Sequential(  # 2 * 1024 * 1024
            nn.Conv2d(2, 3, kernel_size=11, stride=2),  # 3 * 507 * 507
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),  # 3 * 253 * 253
            nn.Conv2d(3, 64, kernel_size=11, stride=4),  # 64 * 60 * 60
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),  # 64 * 30 * 30
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 192 * 30 * 30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),  # 192 * 14 * 14
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 384 * 14 * 14
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256 * 14 * 14
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),  # 256 * 6 * 6
        )
        # 定义回归部分，包含Dropout和全连接层，用于输出最终的回归结果
        self.regression = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(
                in_features=256 * 6 * 6 + num_params, out_features=4096, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2, bias=True),
        )

    def forward(self, surf, p):
        surf = self.features(surf)
        surf = surf.view(surf.size(0), -1)
        p = p.view(p.size(0), -1)
        x = torch.cat((surf, p), 1)
        x = self.regression(x)
        return x
