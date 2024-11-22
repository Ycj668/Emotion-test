import torch
import torch.nn as nn
from torch.nn import init
from utils import knn_value
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size, in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output

class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(

            nn.Linear(lstm_features, node_num * in_features),
            nn.GELU() )

    def forward(self, nodes):
        adj = knn_value(nodes)
        batch_size, window_size, node_num = adj.size()[0: 3]
        eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        adjs = adj + eye
        diag = adjs.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjs.size()) * eye
        adjacency = diag.matmul(adjs).matmul(diag)
        nodes = nodes.permute([0,1,3,2])
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        rout, (_, _) = self.lstm(gcn_output)
        output = rout[:, -1, :]
        output = self.ffn(output)
        return output

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.Sigmoid()
            nn.GELU()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, 4),
            # nn.Sigmoid()
            nn.GELU()  )

    def forward(self, input):
        output = self.ffn2(self.ffn1(input)).squeeze(-1)

        return output

class Classfication(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Classfication, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.bn = nn.BatchNorm1d(self.window_size)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True)
        self.ffn = nn.Sequential(

            nn.Linear(lstm_features, node_num * in_features),
            nn.Linear(node_num * in_features, 2),
            nn.Sigmoid())

    def forward(self, nodes):
        adj = knn_value(nodes)
        batch_size, window_size, node_num = adj.size()[0: 3]
        eye = torch.eye(node_num).to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        adjs = adj + eye
        diag = adjs.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjs.size()) * eye
        adjacency = diag.matmul(adjs).matmul(diag)
        nodes = nodes.permute([0,1,3,2])
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        gcn_output = self.bn(gcn_output)
        rout, (_, _) = self.lstm(gcn_output)
        output = rout[:, -1, :]
        output = self.ffn(output)
        return output

class blk_cha_comb(nn.Module):

    def __init__(self, C, F):
        super(blk_cha_comb, self).__init__()
        self.cov1d = nn.Conv1d(C, C, 1, padding="same")    # c,2c,1
        self.ln = nn.LayerNorm([C, F])    # 2c,f
        self.gelu = nn.GELU()  # 这个可否换成别的激活函数？
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.cov1d(x)
        x = self.ln(x)
        x = self.gelu(x)
        out = self.dropout(x)

        return out

class blk_encoder(nn.Module):

    def __init__(self, C, F, WAY):
        super(blk_encoder, self).__init__()

        self.para_C = C    # 2c
        self.WAY = WAY
        self.C = C
        self.F = F
        self.cnn_ln = nn.LayerNorm([C, F])
        self.cnn_cov1d = nn.Conv1d(C, C, 31, padding="same")
        self.cnn_ln2 = nn.LayerNorm([C, F])
        self.cnn_gelu = nn.GELU()
        self.cnn_dropout = nn.Dropout(0.4)

        # Channel MLP module
        self.mlp_ln = nn.LayerNorm([C, F])
        if self.WAY == 1:
            self.mlp_linear = nn.Linear(F, F)
        elif self.WAY == 2:
            self.mlp_ML_linear = nn.ModuleList([
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F),
                nn.Linear(F, F), nn.Linear(F, F)])
        elif self.WAY == 3:
            self.mlp_WAY3_linear = nn.Linear(2 * C * F, 2 * C * F)

        self.mlp_gelu = nn.GELU()
        self.mlp_dropout = nn.Dropout(0.4)

    def forward(self, x_0):

        x = self.cnn_ln(x_0)
        x = self.cnn_cov1d(x)
        x = self.cnn_ln2(x)
        x = self.cnn_gelu(x)
        x = self.cnn_dropout(x)
        x_1 = x_0 + x
        x = self.mlp_ln(x_1)
        if self.WAY == 1:

            for idx in range(self.para_C):
                y = x[:, idx, :]
                y = self.mlp_linear(y)
                z = torch.unsqueeze(y, 1)
                if idx == 0:
                    zz = z
                else:
                    zz = torch.cat([zz, z], 1)
        elif self.WAY == 2:

            for idx, layer in enumerate(self.mlp_ML_linear):
                y = x[:, idx, :]
                y = layer(y)
                z = torch.unsqueeze(y, 1)
                if idx == 0:
                    zz = z
                else:
                    zz = torch.cat([zz, z], 1)
        elif self.WAY == 3:
            x = torch.flatten(x, 1, 2)
            x = self.mlp_WAY3_linear(x)
            zz = torch.reshape(x, (-1, 2 * self.C, self.F))

        x = self.mlp_gelu(zz)
        x = self.mlp_dropout(x)
        x_3 = x_1 + x
        return x_3
class blk_mlp(nn.Module):

    def __init__(self, C, F, N):
        super(blk_mlp, self).__init__()
        self.para_N = N
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(C*F, 6*N)
        self.ln = nn.LayerNorm(6*N)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(6*N, N)

    def forward(self, x):
        x = torch.flatten(x, 1, 2)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        out = self.linear2(x)

        return out

class mytEEGTransformer(nn.Module):

    def __init__(self, C, F, N, WAY):
        super(mytEEGTransformer, self).__init__()
        self.para_num_fb = 4

        self.blk_cc = blk_cha_comb(C, F)
        self.blk_se_1 = blk_encoder(C, F, WAY)
        self.blk_se_2 = blk_encoder(C, F, WAY)
        self.blk_head = blk_mlp(C, F, 4)
        self.out_linear = nn.Linear(4, 4)

    def forward(self, X):
        features = np.zeros((0, 6, 62))
        for idx in range(self.para_num_fb):
            x = X[:, :, :, idx]
            x = self.blk_cc(x)
            x = self.blk_se_1(x)
            x = self.blk_se_2(x)
            out_tmp1 = self.blk_head(x)
            features = np.vstack((features, x.data.cpu().numpy()))
            if idx == 0:
                out_tmp2 = out_tmp1
            else:
                out_tmp2 = out_tmp2 + out_tmp1
        out = self.out_linear(out_tmp2)

        return out, features
