import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from typing import List, Union, Callable, Optional

from util.model.surf.modified_cnn_model import ModifiedPretrainedNet
from PIL import Image
from abc import ABCMeta, abstractmethod
from .hooked_obj import HookedObj
from sklearn.decomposition import KernelPCA

# TODO
# * 考虑adaptor模块
# * 实现更多预训练网络类型的cam. 目前仅支持resnet和densenet.另外, resnet未进行测试
# * scorecam的实现,如何计算weight
# * 获取模型的各层的方法抽象出来,为其他可视化方法使用
# * 将forward中feature层到classifier层的方法抽象出来,为其他可视化方法使用
class CamBase(HookedObj, metaclass=ABCMeta):
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
        adaptor: Optional[nn.Module] = None,
        target_layer: Optional[Union[str | List[str]]] = None,
        verbose: bool = False,
    ):
        super(CamBase, self).__init__(model, device, verbose)

        self._pretrained_net = self.model.pretrained_net.pretrained_net
        self._feature_layer_dict = OrderedDict()
        self._adaptor = adaptor
        self._set_feature_layer_dict()

        self.__print_target_layer()
        self._target_layer = []
        self.__set_target_layer(target_layer)
        self._gradient = []
        self._conv_output = []
        self._cam = []
        self._cam_fig = []
        self._high_res_cam_fig = []
        self._batch = None

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
        if self._adaptor is not None:
            self._feature_layer_dict = {
                "adaptor": self._adaptor,
                **self._feature_layer_dict,
            }

    def get_feature_layer_dict(self):
        return self._feature_layer_dict

    def __set_target_layer(self, target_layer: Optional[Union[str | List[str]]]):
        if target_layer is not None:
            if isinstance(target_layer, str):
                if target_layer == "all":
                    self._target_layer = list(self._feature_layer_dict.keys())
                    return
                else:
                    target_layer = [target_layer]
            if isinstance(target_layer, list):
                for tag in target_layer:
                    if tag in self._feature_layer_dict:
                        self._target_layer.append(tag)
                    elif self.verbose:
                        print(f"Target layer is not available")
                if len(self._target_layer) > 0:
                    return
        self._target_layer.append(list(self._feature_layer_dict.keys())[-1])
        if self.verbose:
            print(
                f"No available target layer specified, using the last featuring layer [{self._target_layer}]"
            )

    def __print_target_layer(self):
        if self.verbose:
            print(f"Following target layers are available:")
            print(f"{[name for name in self._feature_layer_dict]}")

    def _hook_layers(self):
        def __save_grad_hook(module, input, output):
            def __store_grad(grad):

                self._gradient.insert(0, grad[0].cpu().detach())
                if self.verbose:
                    print(f"Gradient is saved")

            output.register_hook(__store_grad)
            self._conv_output.append(output[0].cpu().detach())
            if self.verbose:
                print(f"Activation is saved")

        for name in self._feature_layer_dict:
            if name in self._target_layer:
                if self.verbose:
                    print(f"Target layer [{name}] found")
                self._handle.append(
                    self._feature_layer_dict[name].register_forward_hook(
                        __save_grad_hook
                    )
                )

    def _get_2d_projection(self, activation):
        # input (C,H,W)
        activation[np.isnan(activation)] = 0
        reshaped_activation = activation.reshape(activation.shape[0], -1).transpose()
        # (C, H, W) -> (H * W, C)
        reshaped_activation -= reshaped_activation.mean(axis=0)
        _, _, VT = np.linalg.svd(reshaped_activation, full_matrices=False)
        projection = reshaped_activation @ VT[0, :]
        projection = projection.reshape(activation.shape[1:])
        return np.float32(projection)

    def __cam_to_fig(self, cam):
        # cam_fig = np.uint8((cam - np.min(cam)) / (np.max(cam) - np.min(cam)) * 255)
        # cam_fig = (
        #     np.uint8(
        #         Image.fromarray(cam_fig).resize(
        #             (self._batch[0].shape[2], self._batch[0].shape[3]), Image.LANCZOS
        #         )
        #     )
        #     / 255
        # )
        # return cam_fig
        img = cam - np.min(cam)
        img = img / (1e-7 + np.max(img))
        img = np.uint8(img * 255)
        img = (
            np.uint8(
                Image.fromarray(img).resize(
                    (self._batch[0].shape[2], self._batch[0].shape[3]), Image.LANCZOS
                )
            )
            / 255
        )
        return img

    @abstractmethod
    def _cam_gen(self, conv, grad):
        pass
    # @HookedObj._hooked
    # def generate_cam_on_fig(self,batch,target,loss_func: nn.Module = nn.MSELoss()):

    @HookedObj._hooked
    def generate_cam(
        self,
        batch,
        loss_func: nn.Module = nn.MSELoss(),
    ):
        self._batch = batch
        #! 重要提示：没有 requires_grad_() 就不会计算梯度
        self._batch = [item.to(self.device).requires_grad_() for item in self._batch]
        targets = self._batch[-1]
        # 前向传播：计算模型的输出
        model_outputs = self.model(*self._batch[0:-1])
        # 将梯度重置为零，为下一次反向传播做准备
        self.model.zero_grad()
        loss = loss_func(model_outputs, targets)
        loss.backward(retain_graph=True)
        for conv, grad in zip(self._conv_output, self._gradient):

            self._cam.append(self._cam_gen(conv, grad))
            self._cam_fig.append(
                self.__cam_to_fig(np.maximum(self._cam[-1].sum(axis=0), 0))
            )
            self._high_res_cam_fig.append(
                self.__cam_to_fig(np.maximum(self._get_2d_projection(self._cam[-1]), 0))
            )
        return (
            self._cam,
            self._conv_output,
            self._gradient,
            self._cam_fig,
            self._high_res_cam_fig,
        )


class EigenCam(CamBase):

    def _cam_gen(self, conv, grad):
        return conv.numpy()


class EigenGradCam(CamBase):
    def _cam_gen(self, conv, grad):
        # weights = np.mean(grad.numpy(), axis=(1, 2))
        # conv = conv.numpy()
        # cam = np.zeros(conv.shape[1:], dtype=np.float32)
        # for i, w in enumerate(weights):
        #     cam += w * conv[i, :, :]
        return grad.numpy() * conv.numpy()


class ElementWiseGradCam(CamBase):

    def _cam_gen(self, conv, grad):
        return np.maximum(grad.numpy() * conv.numpy(), 0)


class GradCam(CamBase):

    def _cam_gen(self, conv, grad):
        weights = np.mean(grad.numpy(), axis=(1, 2))
        return weights[:, None, None] * conv.numpy()


class GradCamPlusPlus(CamBase):

    def _cam_gen(self, conv, grad):
        grad = grad.numpy()
        conv = conv.numpy()
        grad_power_2 = grad**2
        grad_power_3 = grad_power_2 * grad
        sum_activation = np.sum(conv, axis=(1, 2))
        eps = 0.000001
        aij = grad_power_2 / (
            2 * grad_power_2 + sum_activation[:, None, None] * grad_power_3 + eps
        )
        aij = np.where(grad != 0, aij, 0)
        weights = np.maximum(grad, 0) * aij
        weights = np.sum(weights, axis=(1, 2))
        return weights[:, None, None] * conv


class XGradCam(CamBase):
    def _cam_gen(self, conv, grad):
        conv = conv.numpy()
        sum_activation = np.sum(conv, axis=(1, 2))
        eps = 1e-7
        weights = grad.numpy() * conv / (sum_activation[:, None, None] + eps)
        weights = weights.sum(axis=(1, 2))
        return weights[:, None, None] * conv


class KPCACam(CamBase):
    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        adaptor=None,
        target_layer=None,
        verbose=False,
        kernel="sigmoid",
        gamma=None,
    ):
        super().__init__(model, device, adaptor, target_layer, verbose)
        self.__kernel = kernel
        self.__gamma = gamma

    def _cam_gen(self, conv, grad):
        return conv.numpy()

    def _get_2d_projection(self, activation):
        # input (C,H,W)
        activation[np.isnan(activation)] = 0
        reshaped_activation = activation.reshape(activation.shape[0], -1).transpose()
        # (C, H, W) -> (H * W, C)
        reshaped_activation -= reshaped_activation.mean(axis=0)
        kpca = KernelPCA(n_components=1, kernel=self.__kernel, gamma=self.__gamma)
        projection = kpca.fit_transform(reshaped_activation)
        projection = projection.reshape(activation.shape[1:])
        return np.float32(projection)


class ScoreCam(CamBase):

    """
    # ! REFACTOR NEEDED !!! NOT available NOW! DONNOT use it
    """
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

    def _cam_gen(self, conv, grad):
        weights = np.maximum(grad, 0)
        return weights.numpy() * conv.numpy()


class RandomCam(CamBase):

    def _cam_gen(self, conv, grad):
        return conv.numpy() * np.random.uniform(-1, 1, size=(grad.shape))
