import torch
import torch.nn as nn


class TcrPairEmbedding(nn.Module):

    def __init__(
        self,
        n_dim: int,
        size_vocab: int,
        n_prop: int,
        n_features: int,
        num_embeddings: int = 256,
        prop_ratio: float = 0.5,
        feature_ratio: float = 0.5,
        dropout: float = 0.2,
    ):
        super(TcrPairEmbedding, self).__init__()
        prop_out_features = int(num_embeddings * prop_ratio)
        mat_out_features = num_embeddings - prop_out_features
        self.t = nn.Embedding(num_embeddings=size_vocab, embedding_dim=1)
        self.mat_featuring = nn.Sequential(
            # nn.Embedding(num_embeddings=n_dim, embedding_dim=1),
            nn.Linear(in_features=n_dim, out_features=mat_out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(mat_out_features),
        )

        self.prop_featuring = nn.Sequential(
            nn.Linear(in_features=n_prop, out_features=prop_out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(prop_out_features),
        )
        self.embedding = nn.Sequential(
            nn.Linear(in_features=num_embeddings, out_features=num_embeddings),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_embeddings),
        )
        embed_out_features = int(num_embeddings * feature_ratio)
        feature_out_features = num_embeddings - embed_out_features
        self.embed_featuring = nn.Sequential(
            nn.Linear(in_features=num_embeddings, out_features=embed_out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_out_features),
        )
        self.feature_featuring = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=feature_out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_out_features),
        )
        self.regression = nn.Sequential(
            nn.Linear(in_features=num_embeddings, out_features=num_embeddings * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_embeddings * 2),
            nn.Dropout(dropout),
            nn.Linear(in_features=num_embeddings * 2, out_features=1),
        )

    def forward(self, feature_data, prop_data, embed_data):
        m = self.t(embed_data)
        m = m.squeeze(2)
        m = self.mat_featuring(m)
        p = self.prop_featuring(prop_data)
        e = torch.cat((m, p), dim=1)
        f = self.feature_featuring(feature_data)
        e = self.embedding(e)
        e = self.embed_featuring(e)
        x = torch.cat((e, f), dim=1)
        x = self.regression(x)
        return x
