import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Siamese_Reducer(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features=4096, num_features=500):
        super(Siamese_Reducer, self).__init__()
        self.reduce = nn.Linear(in_features, num_features)

    def forward(self, emb_1, emb_2):
        emb1 = self.reduce(emb_1)
        emb2 = self.reduce(emb_2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2
