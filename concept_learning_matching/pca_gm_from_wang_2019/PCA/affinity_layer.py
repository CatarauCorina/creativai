import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        # self.A = Parameter(Tensor(self.d, self.d))
        self.A = torch.nn.Bilinear(self.d, self.d, 15).to(device)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.d)
    #     self.A.data.uniform_(-stdv, stdv)
    #     self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        #M = torch.matmul(X, self.A)
        # M = self.A(X.T)
        # M =
        # M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        # M = torch.matmul(M, Y.transpose(1, 2))
        M = self.A(X,Y)
        return M
