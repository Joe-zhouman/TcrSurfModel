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

util_path = "/root/src"  # path to store the util package
sys.path.insert(0, util_path)  # the util package is supposed to be clone to this path

from util.torch_model.surf_model.modified_cnn_model import (
    ModifiedPretrainedNet,
    SurfNet256,
)
from util.torch_model.surf_model.pretrained_model import PretrainedModelDb
from util.torch_model.surf_dateset import SurfDatasetFromMat

from util.torch_training import cross_validate, get_train_info_logger
from torch import optim

from typing import Dict, Union


def start_train(
    train_type: str = "latest",
    pretrain: bool = False,
    train_model_name: str = "densenet",
    train_model_type: Union[int, str] = "121",
    suffix: str = "input254_cv5",
    dropout: float = 0.2,
    lr: float = 0.001,
    cnn_feature_ratio=0.5,
    epoches: int = 100,
    batch_size: int = 128,
    data_root_path: str = "/hy-tmp/",
    data_csv_filename: str = "DataNormilized.csv",
):
    """
    params:
    - train_type: "start", "best","lastest"
    - pretrain: weather to use pretrained model
    - dropout: dropout rate in regression layer
    - lr: learning rate of optimizer
    - cv: cross validate epoches
    - batch_size: training batch size. 128 is suitable for 4090
    - data_root_path: path to store train data
    - data_csv_filename = "DataNormilized.csv"  # file to store params to be preprocessed
    """
    model_info_db = PretrainedModelDb()
    train_model, model_weights, name_first_conv, name_fc = model_info_db.get_info(
        train_model_name, train_model_type
    )

    model_name = f"{train_model_name}{train_model_type}_{suffix}"

    pnet = ModifiedPretrainedNet(
        pretrained_net=train_model,
        weights=model_weights if pretrain else None,
        name_first_conv=name_first_conv,
        name_fc=name_fc,
    )

    surf_model = SurfNet256(
        modified_net=pnet,
        num_params=3,
        num_output=2,
        dropout=dropout,
        cnn_feature_ratio=cnn_feature_ratio,
    )
    surf_model.to(device)

    dset = SurfDatasetFromMat(
        data_csv_filename=os.path.join(data_root_path, "train", data_csv_filename),
        surf_data_dir=os.path.join(data_root_path, "train", "Surf"),
        param_start_idx=3,
        param_end_idx=6,
        num_targets=2,
    )

    optimizer = optim.Adam(surf_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")
    save_root_path = f"./checkpoint/{model_name}/"

    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    logger = get_train_info_logger(os.path.join(save_root_path, "train_info.log"))
    if not train_type == "start":
        checkpoint_path = os.path.join(
            save_root_path,
            f"{model_name}_{train_type}.ckpt",
        )
        checkpoint = torch.load(checkpoint_path)
        surf_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        kwds = {
            "start_epoches": checkpoint["epoch"],
            "best_loss": checkpoint["best_loss"],
            "loss": checkpoint["loss"],
        }
    else:
        kwds = {}
    cross_validate(
        dataset=dset,
        training_model=surf_model,
        optim=optimizer,
        loss_func=loss_func,
        batch_size=batch_size,
        epoches=epoches,
        model_name=model_name,
        root_path=save_root_path,
        logger=logger,
        **kwds,
    )


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
    start_train(
        train_type=train_type,
        train_model_name="regnet",
        train_model_type="_y_400mf",
        pretrain=args.pretrain in ["t", "true"],
        dropout=0.2,
        lr=0.001,
        epoches=100,
        batch_size=128,
        cnn_feature_ratio=0.5,
        suffix="input254_cv5_train10000",
        data_csv_filename="DataNormilized_2.csv",
    )
