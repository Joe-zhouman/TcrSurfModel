from time import time
from typing import Optional, Union, Tuple, List, Dict
from os import path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn import Module
from torch.optim import Optimizer
from sklearn.model_selection import KFold
import numpy as np


def train_model(
    training_model: Module,
    dloader: dict[str, DataLoader],
    loss_func: Module,
    optimizer: Optimizer,
    root_path: str = ".",
    model_name: Optional[str] = None,
    epoches: int = 10,
    start_epoch: int = 0,
    loss: Dict[str, list] = {"train": [], "val": []},
    best_val_loss:float = float("inf"),
    device: str = "cuda",
):
    """
    训练模型的函数。也可以重启训练。

    参数:
    - training_model: 正在训练的模型。
    - dloader: 包含训练和验证数据的数据加载器字典。需要为以下形式: {"train": train_loader, "val": val_loader}
    - loss_func: 损失函数。
    - optimizer: 优化器。
    - save_path: 模型保存路径,默认为当前目录。
    - save_name: 模型保存名称,默认为模型类的名称。
    - epoches: 训练轮数,默认为10。
    - start_epoch: 从哪个轮数开始训练,默认为0。
    - loss: 存储训练和验证过程的损失值的字典,默认为空需要为以下形式: {"train": [], "val": []}。
    """
    # 如果未提供save_name,则使用模型的类名
    if model_name is None:
        model_name = type(training_model).__name__

    # 计算训练和验证数据集的大小
    dset_size = {s: len(dset) for s, dset in dloader.items() if s in ["train", "val"]}
    
    # 开始训练和验证过程
    for e in range(start_epoch, start_epoch + epoches):
        start_time = time()
        print(f"[{datetime.now()}]: Epoch {e} start")

        # 对于每个epoch,分别处理训练和验证数据集
        for dataset in ["train", "val"]:
            print(f"[{datetime.now()}]: {dataset} start")
            loss_epoch = 0

            # 根据数据集设置模型的训练/评估模式
            if dataset == "train":
                training_model.train()
            else:
                training_model.eval()

            # 遍历数据集中的所有数据
            for surf, para, targets in dloader[dataset]:
                optimizer.zero_grad()

                # 将数据移动到指定设备
                surf = surf.to(device)
                para = para.to(device)
                targets = targets.to(device)

                # 根据当前是否在训练阶段, 决定是否启用梯度计算
                with torch.set_grad_enabled(dataset == "train"):
                    outputs = training_model(surf, para)
                    current_loss = loss_func(outputs, targets)

                    # 在训练阶段,执行反向传播和优化步骤
                    if dataset == "train":
                        current_loss.backward()
                        optimizer.step()

                # 累加当前批次的损失值
                loss_epoch += current_loss.item() * surf.size(0)

            # 计算并存储当前阶段的平均损失值
            loss[dataset].append(loss_epoch / dset_size[dataset])
            print(
                f"[{datetime.now()}]: {dataset} end with loss {loss[dataset][-1]:.4f}"
            )

            # 在验证阶段结束后,保存模型的检查点
            if dataset == "val":
                if loss["val"][-1] < best_val_loss:
                    best_val_loss = loss["val"][-1]
                    save_checkpoints(
                        training_model, 
                        optimizer, 
                        root_path, 
                        model_name, 
                        loss, 
                        e=e,
                        suffix="best",
                        best_loss=best_val_loss
                    )
                save_checkpoints(
                        training_model, 
                        optimizer, 
                        root_path, 
                        model_name, 
                        loss, 
                        e=e,
                        best_loss=best_val_loss
                    )
                end_time = time()
                print(
                    f"[{datetime.now()}]: Epoch {e} end with time {(end_time - start_time)/3600:.4f}"
                )
                print("====================", end="\n")


def save_checkpoints(
    training_model: Module,
    optimizer: Optimizer,
    root_path: str,
    model_name: str,
    loss: Dict[str, float],
    e: int,
    best_loss:float,
    suffix: str = "latest",
):
    """
    保存训练模型的检查点。

    此函数用于保存模型在特定训练阶段的状态,包括模型的参数、优化器的状态、当前的损失值和训练的轮次。
    这使得模型能够在未来的某个时间点继续训练或者进行评估。

    参数:
    - training_model (Module): 当前训练的模型。
    - optimizer (Optimizer): 当前使用的优化器。
    - root_path (str): 保存检查点的根目录路径。
    - model_name (str): 模型的名称,用于生成保存文件的名称。
    - loss (Dict[str, float]): 当前模型的损失值,通常包括一个或多个损失项。
    - e (int): 当前训练的轮次
    - best_loss (float): 所有轮次里的最佳损失值
    - suffix (str): 模型的训练轮次或者标识,用于生成保存文件的名称,默认为"latest"。
    """
    # 生成保存模型检查点的完整路径
    save_path = path.join(root_path, f"surf_{model_name}_{suffix}.ckpt")

    # 保存模型检查点,包括当前训练轮次、模型状态字典、优化器状态字典和损失值
    torch.save(
        {
            "epoch": e,
            "model_state_dict": training_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_loss":best_loss
        },
        save_path,
    )
    print(f"[{datetime.now()}]: Save {suffix} model to {save_path}")


def train_single_fold(
    training_model: Module,
    dloader: dict[str, DataLoader],
    loss_func: Module,
    optimizer: Optimizer,
    device: str = "cuda",
) -> float:
    """
    训练模型的函数。也可以重启训练。

    参数:
    - training_model: 正在训练的模型。
    - dloader: 包含训练和验证数据的数据加载器字典。需要为以下形式: {"train": train_loader, "val": val_loader}
    - loss_func: 损失函数。
    - optimizer: 优化器。
    """
    # 如果未提供save_name,则使用模型的类名
    if model_name is None:
        model_name = type(training_model).__name__
    # 计算训练和验证数据集的大小
    dset_size = {s: len(dset) for s, dset in dloader.items() if s in ["train", "val"]}

    # 开始训练和验证过程

    # 对于每个epoch,分别处理训练和验证数据集
    for dataset in ["train", "val"]:
        print(f"[{datetime.now()}]: {dataset} start")
        loss_epoch = 0

        # 根据数据集设置模型的训练/评估模式
        if dataset == "train":
            training_model.train()
        else:
            training_model.eval()

        # 遍历数据集中的所有数据
        for surf, para, targets in dloader[dataset]:
            optimizer.zero_grad()

            # 将数据移动到指定设备
            surf = surf.to(device)
            para = para.to(device)
            targets = targets.to(device)

            # 根据当前是否在训练阶段, 决定是否启用梯度计算
            with torch.set_grad_enabled(dataset == "train"):
                outputs = training_model(surf, para)
                current_loss = loss_func(outputs, targets)

                # 在训练阶段,执行反向传播和优化步骤
                if dataset == "train":
                    current_loss.backward()
                    optimizer.step()

            # 累加当前批次的损失值
            loss_epoch += current_loss.item() * surf.size(0)

        # 计算并存储当前阶段的平均损失值
        loss = loss_epoch / dset_size[dataset]
        print(f"[{datetime.now()}]: {dataset} end with loss {loss:.4f}")

        # 在验证阶段结束后,返回验证损失值
        if dataset == "val":
            return loss


def cross_validate(
    dataset: Dataset,
    training_model: Module,
    optim: Optimizer,
    loss_func: Module,
    n_splits: int = 5,
    batch_size: int = 128,
    epoches: int = 10,
    start_epoches: int = 0,
    root_path: str = ".",
    model_name: Optional[str] = None,
    loss: Dict[str, float] = {"mean": [], "std": []},
    best_loss:float=float("inf")
):
    """
    对给定的数据集进行交叉验证训练和评估。

    参数:
    - dataset: Dataset 实例,包含所有训练和验证数据。
    - training_model: Module 实例,待训练的模型。
    - optim: Optimizer 实例,用于模型训练的优化器。
    - loss_func: Module 实例,用于计算损失的函数。
    - n_splits: int,默认为5,表示交叉验证的折数。
    - batch_size: int,默认为128,表示每个批次的样本数。
    - epoches: int,默认为10,表示训练的轮数。
    - start_epoches: int,默认为0,表示开始训练的轮数。
    - root_path: str,默认为".",表示保存模型和结果的根路径。
    - model_name: Optional[str],模型的名称,如未提供,则使用模型的类名。
    - loss: Dict[str, float],用于存储每轮训练的损失均值和标准差。
    - best_loss: float, 用于存储最佳损失。
    """
    # 如果未提供save_name,则使用模型的类名
    if model_name is None:
        model_name = type(training_model).__name__
    # 初始化KFold对象进行交叉验证分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # 初始化最佳损失为无穷大
    dloader = []
    for train_idx, val_idx in kf.split(dataset):
        dloader.append(
            {
                "train": DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(train_idx),
                ),
                "val": DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(val_idx),
                ),
            }
        )
    # 遍历每个epoch
    for e in range(start_epoches, start_epoches + epoches):
        start_time = time()
        print(f"[{datetime.now()}]: Epoch {e} start")
        fold_loss = []
        # 遍历每个交叉验证折
        for fold in range(n_splits):
            print(f"[{datetime.now()}]: Fold {fold} start")
            # 创建训练和验证的数据加载器
            # 训练单个折并获取损失
            floss = train_single_fold(
                training_model=training_model,
                dloader=dloader[fold],
                loss_func=loss_func,
                optimizer=optim,
                device="cuda",
            )
            fold_loss.append(floss)
            print(f"[{datetime.now()}]: Fold {fold} end with loss {floss}")
        # 计算当前epoch的损失均值和标准差
        fold_loss_mean = np.mean(fold_loss)
        fold_loss_std = np.std(fold_loss)
        loss["mean"].append(fold_loss_mean)
        loss["std"].append(fold_loss_std)
        end_time = time()
        print(
            f"[{datetime.now()}]: End epoch {e} with {n_splits} folds in {(end_time-start_time)/3600:.4f} hours with loss {fold_loss_mean:.4f} +/- {fold_loss_std:.4f}"
        )
        # 如果当前损失均值为最佳,则保存模型
        if fold_loss_mean < best_loss:
            best_loss = fold_loss_mean
            save_checkpoints(
                training_model=training_model,
                optimizer=optim,
                model_name=model_name,
                root_path=root_path,
                loss=loss,
                e=e,
                suffix="best",
                best_loss=best_loss
            )

        # 保存最新模型
        save_checkpoints(
            training_model=training_model,
            optimizer=optim,
            model_name=model_name,
            root_path=root_path,
            e=e,
            loss=loss,
            best_loss=best_loss
        )
