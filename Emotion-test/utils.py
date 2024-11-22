import torch
import numpy as np
import config as config
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def knn_value(x):
    inner = -2*torch.matmul(x.transpose(3, 2), x)
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(3, 2)
    value = abs(pairwise_distance)
    return value



