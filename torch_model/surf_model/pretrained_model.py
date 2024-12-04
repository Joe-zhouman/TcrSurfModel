import torchvision.models as models
from torch.nn import Module
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union


class PretrainedModelInfo:
    name: str
    """预训练模型名称"""
    cammel_name: str
    """预训练模型名称的驼峰法"""
    first_conv: str
    """预训练模型的第一个卷积层名称"""
    fc: str
    """预训练模型的最后一层全连接层名称"""
    identifiers: List[str]
    """不同大小的预训练模型的标识(放在name后面用于构成预训练模型的类名)"""
    instances: Dict[str, Tuple[Module, models.Weights]]

    def __init__(self) -> None:
        self.instances = {}
        for id in self.identifiers:
            w = getattr(models, f"{self.cammel_name}{id.upper()}_Weights")
            self.instances[id] = (
                getattr(models, f"{self.name}{id}"),
                w.DEFAULT,
            )

    def get_max_level(self) -> int:
        return len(self.identifiers) - 1

    def check_level(self, level: int) -> None:
        max_level = self.get_max_level()
        if level < 0:
            level = max_level + level
        if level > max_level:
            raise ValueError(f"Level {level} is out of range for {self.name}")

    def check_identifier(self, identifier: str) -> None:
        if identifier not in self.identifiers:
            raise ValueError(f"Identifier {identifier} is not in the model")

    def get_identifier(self, level: int) -> str:
        self.check_level(level)
        return self.identifiers[level]

    def get_instance_by_level(self, level: int) -> Tuple[Module, models.Weights]:
        self.check_level(level)
        return self.instances[self.identifiers(level)]

    def get_instance_by_identifier(
        self, identifier: str
    ) -> Tuple[Module, models.Weights]:
        self.check_identifier(identifier)
        return self.instances[identifier]


class DenseNetInfo(PretrainedModelInfo):
    name = "densenet"
    cammel_name = "DenseNet"
    first_conv = "features.conv0"
    fc = "classifier"
    identifiers = ["121", "161", "169", "201"]


class ResNetInfo(PretrainedModelInfo):
    name = "resnet"
    cammel_name = "ResNet"
    first_conv = "conv1"
    fc_name = "fc"
    identifiers = ["18", "34", "50", "101", "152"]


class EfficientNetInfo(PretrainedModelInfo):
    name = "efficientnet"
    cammel_name = "EfficientNet"
    first_conv = "features.0.0"
    fc = "classifier.1"
    identifiers = ["_b0", "_b1", "_b2", "_b3", "_b4", "_b5", "_b6", "_b7"]


class PretrainedModelDb:
    models_list = {
        "densenet": DenseNetInfo,
        "resnet": ResNetInfo,
        "efficientnet": EfficientNetInfo,
    }

    def get_info(
        self, model_name: str, model_type: Union[int | str]
    ) -> Tuple[Module, models.Weights, str, str]:
        instance = self.models_list[model_name]()
        m, w = (
            instance.get_instance_by_level(model_type)
            if type(model_type) is int
            else instance.get_instance_by_identifier(model_type)
        )
        return (
            m,
            w,
            instance.first_conv,
            instance.fc,
        )
