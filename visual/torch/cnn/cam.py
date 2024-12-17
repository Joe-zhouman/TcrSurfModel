import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from typing import List, Callable, Optional

from util.model.surf.modified_cnn_model import ModifiedPretrainedNet
from PIL import Image
from abc import ABCMeta, abstractmethod

class ForwardModelBase(metaclass=ABCMeta):

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.to(self.device)
        self.model.eval()

        self._pretrained_net = self.model.pretrained_net.pretrained_net
        self._pnet_feature_to_classifier_opt = self.__forward()
        self._feature_layer_dict = OrderedDict()

        self._set_feature_layer_dict()
    def __forward(self):
        def __densenet(x: torch.Tensor) -> torch.Tensor:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self._pretrained_net.classifier(x)
            return x

        def __resnet(x: torch.Tensor) -> torch.Tensor:
            x = torch.flatten(x, 1)
            x = self._pretrained_net.fc(x)
            return x

        if isinstance(self._pretrained_net, models.densenet.DenseNet):
            return __densenet
        if isinstance(self._pretrained_net, models.resnet.ResNet):
            return __resnet

    def _set_feature_layer_dict(self):
        if isinstance(self._pretrained_net, models.densenet.DenseNet):
            self._feature_layer_dict = {
                name: module
                for name, module in self._pretrained_net.features.named_children()
            }
        elif isinstance(self._pretrained_net, models.resnet.ResNet):
            self._feature_layer_dict = {
                name: module
                for name, module in self._pretrained_net.named_children()
                if name not in ["fc"]
            }

    def get_feature_layer_dict(self):
        return self._feature_layer_dict

# TODO
# * 考虑adaptor模块
# * 实现更多预训练网络类型的cam. 目前仅支持resnet和densenet.另外, resnet未进行测试
# * scorecam的实现,如何计算weight
# * 获取模型的各层的方法抽象出来,为其他可视化方法使用
# * 将forward中feature层到classifier层的方法抽象出来,为其他可视化方法使用
class CamBase(ForwardModelBase,metaclass=ABCMeta):
    """
    Extracts cam features from the model
    表面模型架构如下:

    ![surf_model架构示意图](https://s2.loli.net/2024/12/17/2xAlQN3Hb7gyjI1.png)

    对应输入为:
    - model:surf_model
    - feature_layers:[surf_model.adaptor(如果有adaptor模块),
    surf_model.pretrained_net.pretrained_net.features(最后的名字可能不一样)]
    #! 注意, 要按顺序传入.
    - pretrained_net_classification_layer:surf_model.pretrained_net.pretrained_net.classifier(最后的名字可能不一样)
    feature和classifier的具体名字可以通过以下代码查看:

    >>> print([name for name, _ in surf_model.pretrained_net.pretrained_net.named_children()])
    #for densenet,output: ['features', 'classifier']

    - target_layer: "denseblock1"(这里以densenet121为例) 即surf_model.pretrained_net.pretrained_net.features下面一个子层的名字.
    可以使用如下代码参考这些层的名字:

    >>> print([name for name, _ in surf_model.pretrained_net.pretrained_net.features.named_children()])
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        super(CamBase, self).__init__(model, device, verbose)

        self._gradients = None

        self.__print_target_layer()

        self._target_layer = None
        self._cam = None
        self._weights = None
        self._conv_outputs = None
        self._batch = None

    def __set_target_layer(self, target_layer: Optional[str]):
        self._target_layer = target_layer
        if self._target_layer is None:
            self._target_layer = list(self._feature_layer_dict.keys())[-1]
            if self.verbose:
                print(
                    f"No target layer specified, using the last featuring layer [{self._target_layer}]"
                )
        self._target_layer = target_layer

    def __print_target_layer(self):
        if self.verbose:
            print(f"Following target layer is available:")
            print(f"{[name for name in self._feature_layer_dict]}")

    def __save_gradient(self, grad):
        self._gradients = grad

    def __forward_pass_on_convolutions(self, x):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for name in self._feature_layer_dict:
            module = self._feature_layer_dict[name]
            x = module(x)
            if name == self._target_layer:
                if self.verbose:
                    print(f"Target layer [{name}] found")
                x.register_hook(self.__save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def _forward_pass(self, surf, params):
        """
        Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.__forward_pass_on_convolutions(surf)
        x = self._pnet_feature_to_classifier_opt(x)
        x = self.model.output(x, params)

        return conv_output, x

    def __backward(self):

        #! 重要提示：没有 requires_grad_() 就不会计算梯度
        self._batch = [item.to(self.device).requires_grad_() for item in self._batch]
        targets = self._batch[-1]
        # 前向传播：计算模型的输出
        conv_outputs, model_outputs = self._forward_pass(*self._batch[0:-1])
        # 将梯度重置为零，为下一次反向传播做准备
        self.model.zero_grad()
        model_outputs.backward(gradient=targets, retain_graph=True)
        self._guided_gradients = self._gradients.data.cpu().numpy()[0]
        self._conv_outputs = conv_outputs.data.cpu().numpy()[0]

    def __cam_to_fig(self):
        self._cam = np.uint8(
            (self._cam - np.min(self._cam))
            / (np.max(self._cam) - np.min(self._cam))
            * 255
        )
        self._cam = (
            np.uint8(
                Image.fromarray(self._cam).resize(
                    (self._batch[0].shape[2], self._batch[0].shape[3]), Image.LANCZOS
                )
            )
            / 255
        )

    @abstractmethod
    def _cam_gen(self):
        pass

    def generate_cam(self, batch, target_layer: Optional[str]):
        self._batch = batch
        self.__set_target_layer(target_layer)
        self.__backward()
        self._cam_gen()
        self.__cam_to_fig()
        return self._cam, self._conv_outputs




class GradCam(CamBase):
    def _cam_gen(self):
        self._weights = np.mean(self._guided_gradients, axis=(1, 2))
        self._cam = np.ones(self._conv_outputs.shape[1:], dtype=np.float32)
        for i, w in enumerate(self._weights):
            self._cam += w * self._conv_outputs[i, :, :]

        self._cam = np.maximum(self._cam, 0)


class ScoreCam(CamBase):
    def _cam_gen(self):
        self._cam = np.ones(self._conv_outputs.shape[1:], dtype=np.float32)
        self._conv_outputs = torch.tensor(
            self._conv_outputs, dtype=torch.float32, device=self.device
        )
        self._weights = torch.zeros(
            self._conv_outputs.shape[0], dtype=torch.float32, device=self.device
        )
        loss_func = nn.MSELoss()
        for i in range(len(self._conv_outputs)):
            saliency_map = torch.unsqueeze(
                torch.unsqueeze(self._conv_outputs[i, :, :], 0), 0
            )
            saliency_map = F.interpolate(
                saliency_map,
                size=(self._batch[0].shape[2], self._batch[0].shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            if saliency_map.max() == saliency_map.min():
                continue
            norm_saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )
            self._batch[0] = self._batch[0] * norm_saliency_map
            _, outputs = self._forward_pass(*self._batch[0:-1])
            #! 评分机制, 需要改进
            # 1. 计算MSELoss
            # 2. 将所有的1 / MSELoss进行softmax归一化
            self._weights[i] = 1.0 / loss_func(outputs, self._batch[-1])
        self._weights = F.softmax(self._weights, dim=0)
        self._weights = self._weights.detach().cpu().numpy()
        self._conv_outputs = self._conv_outputs.detach().cpu().numpy()
        for i, w in enumerate(self._weights):
            self._cam += w * self._conv_outputs[i, :, :]
        self._cam = np.maximum(self._cam, 0)


class LayerCam(CamBase):
    def _cam_gen(self):
        self._weights = self._guided_gradients
        self._weights[self._weights < 0] = 0
        self._cam = np.sum(self._weights * self._conv_outputs, axis=0)
