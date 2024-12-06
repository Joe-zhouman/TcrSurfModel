import torchvision.models as models
import os
import json
from torch.nn import Module
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Callable

UPPER_MODEL_LIST: List[str] = ["regnet", "vgg"]
class PretrainedModelInfo:
    name: str
    """预训练模型名称"""
    cammel_case_name: str
    """预训练模型名称的驼峰法, 用于获取权重"""
    first_conv: str
    """预训练模型的第一个卷积层名称"""
    fc: str
    """预训练模型的最后一层全连接层名称"""
    identifiers: List[str]
    """不同大小的预训练模型的标识(放在name后面用于构成预训练模型的类名)"""
    instances: Dict[str, Tuple[Module, models.Weights]]
    """不同预训练模型的实例和预训练权重"""

    def __init__(self, name: str, config: Dict) -> None:
        """
        初始化模型实例。

        根据提供的配置字典，初始化模型的各个属性，并为每个标识符创建模型实例。

        参数:
        - name (str): 模型名称的字符串，用于构建模型实例。
        - config (Dict): 包含模型配置的字典，包括模型的各种属性和设置。

        返回:
        - None
        """
        self.name = name
        self.cammel_case_name = config["cammel_case_name"]
        self.first_conv = config["first_conv"]
        self.fc = config["fc"]
        self.identifiers = config["identifiers"]
        # 初始化模型实例的字典
        self.instances = {}
        # 遍历每个标识符，创建并存储模型实例
        for id in self.identifiers:
            # 动态获取模型的权重类
            w = getattr(
                models,
                f"{self.cammel_case_name}{id.upper() if self.name in UPPER_MODEL_LIST else id.title()}_Weights",
            )
            # 动态获取模型类和默认权重，并存储为实例属性
            self.instances[id] = (
                getattr(models, f"{self.name}{id}"),
                w.DEFAULT,
            )

    def get_max_level(self) -> int:
        """
        获取最大层级

        此方法用于计算当前对象的最大层级它通过返回标识符列表的长度减一来确定最大层级
        最大层级表示了该类神经网络从小到大的深度层级.

        Returns:
            int: 最大层级，神经网络的深度层级
        """
        return len(self.identifiers) - 1

    def check_level(self, level: int) -> None:
        """
        检查给定的层级是否在允许的范围内。

        此方法首先获取最大层级，然后检查给定的层级是否小于0或超过最大层级。
        如果层级小于0，它会通过将其与最大层级相加来计算一个相对的层级值。
        如果层级超过最大层级，则抛出一个ValueError异常。

        参数:
        - level (int): 需要检查的层级。
        """
        # 获取当前实例的最大层级
        max_level = self.get_max_level()

        # 如果给定的层级小于0，计算绝对层级. 如-1表示最后一级
        if level < 0:
            level = max_level + level

        # 如果计算后的层级大于最大层级，抛出异常
        if level > max_level:
            raise ValueError(f"Level {level} is out of range for {self.name}")

    def check_identifier(self, identifier: str) -> None:
        """
        检查标识符是否存在于模型中。

        如果标识符不在模型中，则抛出一个 ValueError 异常，指出标识符未找到。

        参数:
        identifier (str): 需要检查的标识符。
        """
        # 检查给定的标识符是否在模型的标识符列表中
        if identifier not in self.identifiers:
            # 如果标识符不在列表中，抛出 ValueError 异常
            raise ValueError(f"Identifier {identifier} is not in the model")

    def get_identifier(self, level: int) -> str:
        """
        根据给定的级别获取对应的标识符。

        该方法首先调用check_level方法来验证输入的级别是否有效，然后从identifiers列表中返回对应级别的标识符。

        参数:
        level (int): 需要获取标识符的级别。

        返回:
        str: 对应级别的标识符。

        Raises:
        如果级别无效，check_level方法将抛出异常。
        """
        # 验证输入的级别是否有效
        self.check_level(level)
        # 返回对应级别的标识符
        return self.identifiers[level]

    def get_instance_by_level(self, level: int) -> Tuple[Module, models.Weights]:
        """
        根据给定的等级获取对应的实例。

        该方法首先检查给定的等级是否有效，然后根据等级返回预先定义的实例之一。
        此方法强调了等级与实例之间的映射关系，确保了实例的获取是基于有效的等级进行的。

        参数:
        - level (int): 要获取实例的等级。

        返回:
        - Tuple[Module, models.Weights]: 返回一个元组，包含Module实例和对应的权重信息。
        """
        # 检查等级的有效性
        self.check_level(level)
        # 根据等级返回对应的实例
        return self.instances[self.identifiers(level)]

    def get_instance_by_identifier(
        self, identifier: str
    ) -> Tuple[Module, models.Weights]:
        """
        根据标识符获取实例.

        该方法首先验证给定的标识符是否有效，然后根据验证过的标识符从实例字典中获取对应的模块和权重信息.

        参数:
        - identifier (str): 唯一标识一个实例的标识符.

        返回:
        - Tuple[Module, models.Weights]: 一个元组，包含模块及其对应的权重信息.

        抛出:
        - 如果标识符不在实例字典中，可能会抛出KeyError.
        """
        # 验证标识符的有效性
        self.check_identifier(identifier)
        # 从实例字典中获取并返回模块和权重信息
        return self.instances[identifier]


class PretrainedModelDb:
    """
    初始化模型配置器类，负责解析配置文件并初始化预训练模型信息。

    该类在初始化时接受一个配置文件名参数，默认为"pretrained_models.json"。
    它会根据配置文件中的内容，初始化一个包含所有预训练模型信息的字典。

    参数:
    config_file_name (str): 配置文件的名称，默认为"pretrained_models.json"。
    """
    # ! TODO: 将预训练模型按权重升序排列，方便后续使用
    def __init__(self, config_file_name: str = "pretrained_models.json") -> None:
        self.models_list = {}
        config_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_file_name
        )
        config_dict = self.get_config_from_file(config_file_path)
        for model_name, model_info in config_dict.items():
            self.models_list[model_name] = PretrainedModelInfo(model_name, model_info)

    def get_config_from_file(self, config_file):
        """
        从指定的配置文件中读取并解析配置信息。

        此函数尝试从给定的文件路径中读取配置信息，并将配置信息解析为JSON格式的字典。
        如果配置文件不存在或配置文件的内容不是有效的JSON格式，函数将抛出相应的异常。

        参数:
        config_file (str): 配置文件的路径。

        返回:
        dict: 解析后的配置信息字典。

        异常:
        FileNotFoundError: 如果配置文件不存在。
        ValueError: 如果配置文件的内容不是有效的JSON格式。
        """
        # 检查配置文件是否存在
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"{config_file} is not found")

        try:
            # 打开配置文件并解析JSON内容
            with open(config_file, "r") as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            # 如果解析JSON失败，抛出ValueError异常
            raise ValueError(f"{config_file} is not a valid json file") from e

        # 返回解析后的配置字典
        return config_dict

    def get_info(
        self, model_name: str, model_type: Union[int | str]
    ) -> Tuple[Module, models.Weights, str, str]:
        """
        根据模型名称和类型获取模型信息。

        该方法通过模型名称从内部列表中查找对应的模型实例，并根据提供的模型类型
        （可以是整数级别或字符串标识符）获取具体的模型实例及其权重。此外，还返回
        与模型相关的第一卷积层和全连接层的名称。

        参数:
        - model_name (str): 模型的名称。
        - model_type (Union[int, str]): 模型的类型，可以是整数表示的级别或字符串表示的标识符。

        返回:
        Tuple[Module, models.Weights, str, str]: 包含模型实例、模型权重、第一卷积层名称和全连接层名称的元组。
        """
        # 通过模型名称从列表中获取模型实例
        instance = self.models_list[model_name]

        # 根据模型类型是整数还是字符串，调用相应的方法获取模型实例和权重
        vision_model, weights = (
            instance.get_instance_by_level(model_type)
            if type(model_type) is int
            else instance.get_instance_by_identifier(model_type)
        )

        # 返回模型实例、权重、第一卷积层和全连接层的名称
        return (
            vision_model,
            weights,
            instance.first_conv,
            instance.fc,
        )
