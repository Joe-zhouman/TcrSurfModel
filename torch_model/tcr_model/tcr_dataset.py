import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict


class PairEmbeddingDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        mat_embedding_dict: Dict[str, torch.tensor],
        feature_cols: List[str],
        property_cols: List[str],
    ):
        self.X = X
        self.y = y
        self.dict = mat_embedding_dict
        self.feature_cols = feature_cols
        self.property_cols = property_cols

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sub_df = self.X.iloc[idx, :]
        if len(sub_df.shape) != 1:

            feature_data = torch.tensor(
                sub_df.loc[:, self.feature_cols].values.astype(float),
                dtype=torch.float32,
            ).squeeze(0)
            property_data = torch.tensor(
                sub_df.loc[:, self.property_cols].values.astype(float),
                dtype=torch.float32,
            ).squeeze(0)

            tensors = []
            for row in self.X.itertuples():
                tensors.append(
                    torch.cat(
                        (self.dict[row.mat1], self.dict[row.mat2]),
                        dim=0,
                    )
                )

            embed_data = torch.stack(tensors, dim=1).squeeze(0)
        else:
            feature_data = torch.tensor(
                sub_df.loc[self.feature_cols].values.astype(float),
                dtype=torch.float32,
            ).squeeze(0)
            property_data = torch.tensor(
                sub_df.loc[self.property_cols].values.astype(float),
                dtype=torch.float32,
            ).squeeze(0)
            embed_data = torch.cat(
                (self.dict[sub_df["mat1"]], self.dict[sub_df["mat2"]]),
                dim=0,
            ).squeeze(0)
        target_data = torch.tensor(
            self.y.iloc[idx],
            dtype=torch.float32,
        )
        return feature_data, property_data, embed_data, target_data
