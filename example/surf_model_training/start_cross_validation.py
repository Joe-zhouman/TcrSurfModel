import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from torch import nn
from torchvision import models
import copy
device = torch.device("cuda")
# %env CUBLAS_WORKSPACE_CONFIG=:4096:8
os.environ["TORCH_HOME"]="."
import sys
# torch.cuda.is_available()

from torchvision import models

########################################
# parameters to be set in training 

dropout = 0.2       # dropout rate in regression layer
lr = 0.001          # learning rate of optimizer
cv_epoches = 100    # cross validate epoches

########################################

############################################
# parameters to be set for encironment 
# NOT NEED to modify if you have the setting

data_root_path = "/hy-tmp/" # path to store train data
data_csv_filename = "DataNormilized.csv" # file to store params to be preprocessed
util_path = "/root/util"    # path to store the util package
batch_size = 128 # suitable for 4090
############################################

# add local dir to sys path
sys.path.insert(0, util_path) # the util package is supposed to be clone to this path
from util.torch_model.surf_model.modified_cnn_model import (
    ModifiedPretrainedNet,
    SurfNet256,
)

###################################################
# need to be modified for different pretrained net
model_name = "densenet121_input254_cv5-2"

pnet = ModifiedPretrainedNet(
    pretrained_net=models.densenet121,
    weights=None,
    name_first_conv="features.conv0",
    name_fc="classifier",
)
# model_name = "efficientnetb0_input254_cv5"

# pnet = ModifiedPretrainedNet(
#     pretrained_net=models.efficientnet_b0,
#     weights=None,
#     name_first_conv="features.0.0",
#     name_fc="classifier.1",
# )
###################################################

surf_model = SurfNet256(
    modified_net=pnet, 
    num_params=3, 
    num_output=2, 
    dropout=dropout
    )
# print(surf_model)
surf_model.to(device)

from util.torch_model.surf_dateset import SurfDatasetFromMat

dset = SurfDatasetFromMat(
        data_csv_filename=os.path.join(data_root_path, "train", data_csv_filename),
        surf_data_dir=os.path.join(data_root_path, "train", "Surf"),
        param_start_idx=3,
        param_end_idx=6,
        num_targets=2
    )

from util.torch_training import cross_validate, get_train_info_logger
from torch import optim

optimizer = optim.Adam(surf_model.parameters(), lr=lr)
loss_func = nn.MSELoss()


if not os.path.exists("./checkpoint"):
    os.mkdir("./checkpoint")
save_root_path = f"./checkpoint/{model_name}/"

if not os.path.exists(save_root_path):
    os.mkdir(save_root_path)

logger = get_train_info_logger(os.path.join(save_root_path, "train_info.log"))

cross_validate(
    dataset=dset,
    training_model=surf_model,
    optim=optimizer,
    loss_func=loss_func,
    batch_size=batch_size,
    epoches=cv_epoches,
    model_name=model_name,
    root_path=save_root_path,
    logger=logger,
)
