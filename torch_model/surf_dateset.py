import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from os import path
from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable, List, Optional, Dict
from scipy.io import loadmat


class SurfDataset(Dataset, metaclass=ABCMeta):
    """
    定义一个数据集类，用于处理和加载表面相关的数据。

    参数:
    - data_csv_filename: 包含数据集信息的CSV文件路径。
    - surf_data_dir: 存储表面数据的目录路径。
    - param_start_idx: 参数在csv文件中的起始索引,左闭右开
    - param_end_idx: 参数在csv文件中的结束索引,左闭右开
    - num_targets: 数据集回归目标的数量,为1或2。

    说明:

    - 其中数据文件夹组织为:

        - rootpath
        - |__surf_data_dir/
        - |   |__<surf_data>*
        - |__data_csv_filename [must be csv file]

    - data_csv_filename 必须为如下形式:
        最开始几列为表面的文件名(需要实现get_surf_data方法从文件获取数据),
        中间的[param_start_idx, param_end_idx)为输入参数,
        倒数第二列为相对接触面积,
        最后一列为接触热阻.

        |0 to s        |   ...   |param_start_idx to (param_end_idx-1)|   ...   |area|tcr|

        |surf_filename*|some data|param_data                          |some data|area|tcr|

    - 写出如下形式的原因是:
        1. 目前输出了分型参数 D和G, 这些参数是否作为模型参数还在待定中.
        2. 对于回归模型, 对TCR进行回归还是对Area和TCR同时进行回归还在待定中.
    """
    def __init__(self, data_csv_filename: str, surf_data_dir: str,param_start_idx:int,param_end_idx:int,num_targets:int):
        """
        初始化数据集类，加载CSV数据文件。
        """
        # 读取CSV文件到DataFrame
        self.data_frame = pd.read_csv(data_csv_filename)
        # 存储冲浪数据目录路径
        self.surf_data_dir = surf_data_dir
        self.param_start_idx = param_start_idx
        self.param_end_idx = param_end_idx
        self.num_targets = num_targets

    def __len__(self):
        """
        实现len方法，返回数据集的大小。

        返回:
        - 数据集的样本数量。
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        实现索引方法，根据索引加载和返回数据。

        参数:
        - idx: 样本索引，可以是单个索引或索引列表。

        返回:
        - 一个元组，包含表面数据、参数数据和目标数据。
        """
        # 如果索引是张量，将其转换为列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        surf_data_1, surf_data_2 = self.get_surf_data()

        # 将表面数据转换为张量并堆叠，准备作为输入数据 # 2*1024*1024
        surf_data = torch.stack(
            (
                (torch.from_numpy(surf_data_1[np.newaxis, ...])).float(),
                (torch.from_numpy(surf_data_2[np.newaxis, ...])).float(),
            ),
            dim=1,
        ).squeeze(0)

        # 读取参数数据，转换为张量
        para_data = torch.tensor(
            self.data_frame.iloc[idx, self.param_start_idx:self.param_end_idx].values.astype(float),
            dtype=torch.float32,
        ).unsqueeze(0)

        # 读取预测目标数据，转换为张量 (真实接触面积, 接触热阻)
        target_data = torch.tensor(
            self.data_frame.iloc[idx, -self.num_targets:].values.astype(float),
            dtype=torch.float32,
        )
        if self.num_targets == 1:
            target_data = target_data.unsqueeze(0)
        else:
            target_data = target_data.squeeze(0)
        # 返回表面数据、参数数据和目标数据
        return surf_data, para_data, target_data

    @abstractmethod
    def get_surf_data(self,idx)->Tuple[np.ndarray,np.ndarray]:
        pass


class SurfDatasetFromMat(SurfDataset):
    """
    从mat文件中读取表面数据\n\n{}
    """.format(
        SurfDataset.__doc__
    )

    def get_surf_data(self,idx) -> Tuple[np.ndarray]:
        surf_filepath = path.join(self.surf_data_dir, self.data_frame.iloc[idx, 0])

        # 读取表面数据，转换为numpy数组
        return loadmat(surf_filepath)["Z1"],loadmat(surf_filepath)["Z2"]


class SurfDatasetFromCSV(SurfDataset):
    """
    从csv文件中读取表面数据(老版本)\n\n{}
    """.format(
        SurfDataset.__doc__
    )
    
    def get_surf_data(self,idx) -> Tuple[np.ndarray]:

        # 构造表面数据文件的路径
        surf_filepath_1 = path.join(self.surf_data_dir, self.data_frame.iloc[idx, 0])
        surf_filepath_2 = path.join(self.surf_data_dir, self.data_frame.iloc[idx, 1])

        # 读取表面数据，转换为numpy数组
        return pd.read_csv(surf_filepath_1, header=None).values, pd.read_csv(surf_filepath_2, header=None).values
