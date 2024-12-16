import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from torch import nn
from torchvision import models
import copy

device = torch.device("cuda")
os.environ["TORCH_HOME"] = "."
import sys

util_path = "../../.."  # path to store the util package
sys.path.insert(0, util_path)  # the util package is supposed to be clone to this path

from util.model.surf.modified_cnn_model import (
    ModifiedPretrainedNet,
    SurfNet256,
    SurfNet1024,
)
from util.model.surf.pretrained_model import PretrainedModelDb
from util.model.surf.dateset import SurfDatasetFromMat

from util.train.torch.iteration import cross_validate, get_train_info_logger
from torch import optim

from typing import Dict, Union, Any


def start_train(
    train_model: nn.Module,
    dset: Dataset,
    optimizer: optim,
    loss_func: nn.Module,
    prefix: str,
    train_type: str = "latest",
    epoches: int = 100,
    batch_size: int = 128,
):
    """
    开始训练模型。

    参数:
    - train_model: 待训练的模型。
    - dset: 训练数据集。
    - optimizer: 优化器。
    - loss_func: 损失函数。
    - prefix: 保存文件名前缀。
    - train_type: 训练类型，默认为"latest"。
    - epoches: 训练轮数，默认为100轮。
    - batch_size: 批处理大小，默认为128。
    """
    # 检查并创建checkpoint目录
    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")
    save_root_path = f"./checkpoint/{prefix}/"

    # 检查并创建模型保存的子目录
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    # 初始化日志记录器
    logger = get_train_info_logger(os.path.join(save_root_path, "train_info.log"))

    # 获取最新checkpoint的路径
    latest_checkpoint_path = os.path.join(
        save_root_path,
        f"{prefix}_latest.ckpt",
    )

    # 如果最新checkpoint路径已存在，则发出警告并退出
    if os.path.exists(latest_checkpoint_path):
        # 报警
        print(f"{latest_checkpoint_path} exists, please check.")
        return

    # 根据train_type加载模型和优化器的状态
    if not train_type == "start":
        checkpoint_path = os.path.join(
            save_root_path,
            f"{prefix}_{train_type}.ckpt",
        )
        checkpoint = torch.load(checkpoint_path)
        train_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        kwds = {
            "start_epoches": checkpoint["epoch"] + 1,
            "best_loss": checkpoint["best_loss"],
            "loss": checkpoint["loss"],
        }
    else:
        kwds = {}

    # 执行交叉验证训练
    cross_validate(
        dataset=dset,
        training_model=train_model,
        optim=optimizer,
        loss_func=loss_func,
        batch_size=batch_size,
        epoches=epoches,
        prefix=prefix,
        root_path=save_root_path,
        logger=logger,
        **kwds,
    )


def define_surf_model(
    model_name: str,
    model_type: Union[int, str],
    suffix: str,
    pretrain: bool = True,
    set_type: str = "train",
    kwd: Dict[str, Any] = {
        "dropout": 0.2,
        "cnn_feature_ratio": 0.5,
        "data_root_path": "/hy-tmp/",
        "data_csv_filename": "DataNormilized.csv",
        "lr": 0.001,
        "num_params": 3,
        "num_output": 2,
    },
):
    """
    定义和配置一个用于表面模型的深度学习模型。

    参数:
    - model_name: 模型的名称。
    - model_type: 模型的类型，可以是整数或字符串。
    - suffix: 附加在模型名称和类型之后的后缀，用于区分不同的模型配置。
    - pretrain: 是否使用预训练的模型权重，默认为True。
    - kwd: 包含模型训练和配置所需的各种参数的字典。

    返回:
    一个包含模型配置和训练所需信息的字典，包括模型前缀、模型实例、数据集、优化器和损失函数。
    """

    # 初始化预训练模型数据库
    model_info_db = PretrainedModelDb()

    # 从数据库中获取指定模型和类型的信息
    train_model, model_weights, name_first_conv, name_fc = model_info_db.get_info(
        model_name, model_type
    )

    # 构造模型前缀，用于后续的日志记录或模型保存
    prefix = f"{model_name}{model_type}_{suffix}"

    # 创建一个修改过的预训练网络实例
    pnet = ModifiedPretrainedNet(
        pretrained_net=train_model,
        weights=model_weights if pretrain else None,
        name_first_conv=name_first_conv,
        name_fc=name_fc,
    )

    # 创建并配置最终的表面模型
    surf_model = SurfNet1024(
        modified_net=pnet,
        num_params=kwd["num_params"],
        num_output=kwd["num_output"],
        dropout=kwd["dropout"],
        cnn_feature_ratio=kwd["cnn_feature_ratio"],
    )

    # 将模型移动到指定的设备上
    surf_model.to(device)

    # 创建并配置数据集
    dset = SurfDatasetFromMat(
        data_csv_filename=os.path.join(
            kwd["data_root_path"], set_type, kwd["data_csv_filename"]
        ),
        surf_data_dir=os.path.join(kwd["data_root_path"], set_type, "Surf"),
        param_start_idx=3,
        param_end_idx=6,
        num_targets=2,
    )

    # 创建优化器，用于模型参数的更新
    optimizer = optim.Adam(surf_model.parameters(), lr=kwd["lr"])

    # 定义损失函数，用于衡量模型预测值与真实值的差异
    loss_func = nn.MSELoss()

    # 返回包含模型配置和训练所需信息的字典
    return {
        "prefix": prefix,
        "train_model": surf_model,
        "dset": dset,
        "optimizer": optimizer,
        "loss_func": loss_func,
    }


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "-t",
        "--train_type",
        type=str,
        default="s",
        choices=["s", "start", "l", "latest", "b", "best"],
        help="""
        select train type:
        [s, start]--strat a new train;
        [l, latest]--resume a previous train from the latest checkpoint\n
        [b,best]-- resume a previous train from the best checkpoint\n
        """,
    )
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     nargs=2,
    #     metavar=("model_type", "model_type"),
    #     type=str,
    #     default=["densenet","121"],
    #     action="store",
    #     help="""
    #     select model type:
    #     [resnet, (34,...)]--resnet model;
    #     [densenet, (121, ...)]--densenet model;
    #     [efficientnet, (_b0,...)]--efficientnet model;
    #     other--custom model.

    #     currently only resnet, densenet, efficientnet are supported.
    #     """,
    # )
    parser.add_argument(
        "-p",
        "--pretrain",
        type=str,
        choices=["t", "true", "f", "false"],
        default="f",
        help="""
        use pretrained model or not: 
        t(true) for True, f(false) for False.
        """,
    )

    args = parser.parse_args()
    train_type = args.train_type
    if train_type == "s":
        train_type = "start"
    elif train_type == "l":
        train_type = "latest"
    elif train_type == "b":
        train_type = "best"
    kwd = {
        "dropout": 0.2,
        "cnn_feature_ratio": 0.5,
        "data_root_path": "/hy-tmp/1024/",
        "data_csv_filename": "DataNormilized.csv",
        "lr": 0.001,
        "num_params": 3,
        "num_output": 2,
    }
    model_info = define_surf_model(
        model_name="densenet",
        model_type="121",
        pretrain=args.pretrain in ["t", "true"],
        suffix="input1024_cv5_train10000",
        kwd=kwd,
    )
    start_train(
        train_type=train_type,
        batch_size=128,
        epoches=100,
        **model_info,
    )
