from collections import OrderedDict
import numpy as np
from typing import Tuple, List
import torch.nn as nn

def get_positive_negative_saliency(
    gradient: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据梯度生成正负显著图

    参数:
        gradient (numpy arr): 要可视化的操作的梯度

    返回:
        pos_saliency (bool arr): 正显著图的布尔数组,即结果里所有大于0的元素
        neg_saliency (bool arr): 负显著图的布尔数组,即结果里所有小于0的元素
    """
    pos_saliency = np.maximum(0, gradient) / gradient.max()
    neg_saliency = np.maximum(0, -gradient) / -gradient.min()
    return pos_saliency, neg_saliency






