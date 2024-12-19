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
        self._feature_layer_dict = OrderedDict()

        self._set_feature_layer_dict()

    def _set_feature_layer_dict(self):

        if isinstance(self._pretrained_net, models.resnet.ResNet):
            self._feature_layer_dict = {
                name: module
                for name, module in self._pretrained_net.named_children()
                if name not in ["fc"]
            }
        elif isinstance(self._pretrained_net, models.densenet.DenseNet):
            self._feature_layer_dict = {
                name: module
                for name, module in self._pretrained_net.features.named_children()
            }
        else:
            self._feature_layer_dict = {
                name: module
                for name, module in self._pretrained_net.features.named_children()
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

        self._gradient = None

        self.__print_target_layer()

        self._target_layer = None
        self._cam = None
        self._weights = None
        self._conv_output = None
        self._batch = None
        self._handle = []

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
            print(f"Following target layers are available:")
            print(f"{[name for name in self._feature_layer_dict]}")

    def __hook_layers(self):
        def __save_grad_hook(module, input, output):
            def __store_grad(grad):

                self._gradient = grad[0].cpu().detach()
                if self.verbose:
                    print(f"Gradient is saved")

            output.register_hook(__store_grad)
            self._conv_output = output[0].cpu().detach()
            if self.verbose:
                print(f"Activation is saved")

        for name in self._feature_layer_dict:
            if name == self._target_layer:
                if self.verbose:
                    print(f"Target layer [{name}] found")
                self._handle.append(
                    self._feature_layer_dict[name].register_forward_hook(
                        __save_grad_hook
                    )
                )

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

    def generate_cam(
        self,
        batch,
        loss_func: nn.Module = nn.MSELoss(),
        target_layer: Optional[str] = None,
    ):
        self._batch = batch
        self.__set_target_layer(target_layer)
        self.__hook_layers()
        #! 重要提示：没有 requires_grad_() 就不会计算梯度
        self._batch = [item.to(self.device).requires_grad_() for item in self._batch]
        targets = self._batch[-1]
        # 前向传播：计算模型的输出
        model_outputs = self.model(*self._batch[0:-1])
        # 将梯度重置为零，为下一次反向传播做准备
        self.model.zero_grad()
        loss = loss_func(model_outputs, targets)
        loss.backward(retain_graph=True)
        self._cam_gen()
        self.__cam_to_fig()
        self.release_hook()
        return self._cam

    def release_hook(self):
        for handle in self._handle:
            handle.remove()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release_hook()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True


class GradCam(CamBase):
    def _cam_gen(self):
        weights = np.mean(self._gradient.numpy(), axis=(1, 2))
        self._conv_output = self._conv_output.numpy()
        self._cam = np.ones(self._conv_output.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            self._cam += w * self._conv_output[i, :, :]

        self._cam = np.maximum(self._cam, 0)


class ScoreCam(CamBase):
    def _cam_gen(self):
        self._cam = np.ones(self._conv_output.shape[1:], dtype=np.float32)
        weights = torch.zeros(
            self._conv_output.shape[0], dtype=torch.float32, device=self.device
        )
        _conv_output = self._conv_output.to(self.device)
        loss_func = nn.MSELoss()
        self.release_hook()
        for i in range(len(_conv_output)):
            saliency_map = torch.unsqueeze(torch.unsqueeze(_conv_output[i, :, :], 0), 0)
            saliency_map = F.interpolate(
                saliency_map,
                size=(self._batch[0].shape[2], self._batch[0].shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            if saliency_map.max() == saliency_map.min():
                continue
            saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )
            self._batch[0] = self._batch[0] * saliency_map
            outputs = self.model(*self._batch[0:-1])
            self.model.zero_grad()
            #! 评分机制, 需要改进
            # 1. 计算MSELoss
            # 2. 将所有的1 / MSELoss进行softmax归一化
            weights[i] = 1.0 / loss_func(outputs, self._batch[-1])
        weights = F.softmax(weights, dim=0)
        weights = weights.detach().cpu().numpy()
        _conv_output = _conv_output.detach().cpu().numpy()
        for i, w in enumerate(weights):
            self._cam += w * _conv_output[i, :, :]
        self._cam = np.maximum(self._cam, 0)


class LayerCam(CamBase):
    def _cam_gen(self):
        weights = np.maximum(self._gradient, 0)
        self._cam = np.sum(weights.numpy() * self._conv_output.numpy(), axis=0)
