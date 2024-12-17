"""
@author: Joe-ZhouMan
"""

import torch
import torch.nn as nn

from typing import List
from abc import ABCMeta


class BackpropBase(metaclass=ABCMeta):

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        self.device = device
        # 将模型移动到指定设备
        self.model = model.to(self.device)
        # 初始化梯度属性为None，稍后将通过前向传播计算梯度
        self.gradients = None
        # 将模型设置为评估模式，以关闭Dropout等仅在训练时需要的特性
        self.model.eval()
        # 在目标层上挂载钩子，以提取所需的中间层输出
        self.hook_first_conv_layer()
        self.verbose = verbose

    def hook_first_conv_layer(self):
        """
        在第一层卷积层上挂载钩子，以获取反向传播的最终输出.
        """

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if self.verbose:
                    print(f"conv layer [{name}] is hooked!")
                module.register_full_backward_hook(hook_function)
                # 只需第一个卷积层
                break

    def generate_gradients(self, batch, loss_func):
        """
        计算模型参数相对于损失的梯度。

        该函数处理一批数据，使用提供的损失函数计算损失，然后计算模型参数相对于此损失的梯度。
        参数:
        - batch: 包含输入数据和目标标签的列表。最后一个元素为目标标签。#! 注意: batch_size 必须为1
        - loss_func: 用于计算模型输出与目标标签之间损失的损失函数。
        返回:
        - gradients_as_arr: 包含模型参数梯度的数组。
        """

        # 准备批数据，启用每个项的梯度计算
        batch = [item.requires_grad_().to(self.device) for item in batch]
        #! 重要提示：没有 requires_grad_() 就不会计算梯度
        # 特别确保批数据中的第一项启用了梯度计算，以防在上面的循环中遗漏
        batch[0] = batch[0].requires_grad_()
        # 从批数据中提取目标标签
        targets = batch[-1]
        # 前向传播：计算模型的输出
        outputs = self.model(*batch[0:-1])
        # 将梯度重置为零，为下一次反向传播做准备
        self.model.zero_grad()
        # 计算当前损失
        current_loss = loss_func(outputs, targets)
        # 打印当前损失，以便在训练过程中进行监控
        print(current_loss)
        # 反向传播：计算梯度
        current_loss.backward()
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr


class VanillaBackprop(BackpropBase):
    """
    普通反向传播
    """

    pass


class GuidedBackprop(BackpropBase):
    """
    导向反向传播
    """

    def __init__(
        self,
        model: nn.Module,
        relu_modules: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        super(GuidedBackprop, self).__init__(model, device, verbose)
        """
        初始化模型解释器。

        参数:
        - model: nn.Module，待解释的神经网络模型。
        - relu_modules: List[nn.Module]，包含需要被替换为支持梯度计算的ReLU模块的列表。
        - device: str，模型和数据所使用的计算设备，默认为CUDA（如果可用），否则为CPU。
        """
        # 初始化前向传播ReLU输出的列表，用于存储中间激活值
        self.forward_relu_outputs = []
        # 更新模型中的ReLU模块，以支持保存前向传播过程中的激活值
        self.update_relus(relu_modules)

    def update_relus(self, relu_modules: List[nn.Module]):
        """
        在指定模块中的所有ReLU层上挂载钩子.
        参数:
        - relu_modules: List[nn.Module]，包含需要被替换为支持梯度计算的ReLU模块的列表。
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            在反向传播时对ReLU层进行修改:
            1. 将负值设置为0.
            2. 将正值设置为1.
            """
            corresponding_forward_output = self.forward_relu_outputs.pop()
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            存储ReLU层在前向传播中的输出.
            """
            self.forward_relu_outputs.append(ten_out)

        # 在模型中查找所有ReLU模块并挂载钩子.
        for parent_module in relu_modules:
            for name, module in parent_module.named_modules():
                if isinstance(module, nn.ReLU):
                    if self.verbose:
                        print(f"ReLU layer [{name}] is hooked!")
                    module.register_full_backward_hook(relu_backward_hook_function)
                    module.register_forward_hook(relu_forward_hook_function)
