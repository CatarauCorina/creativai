import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from concept_learning_matching.utils.sinkhorn import Sinkhorn
from concept_learning_matching.utils.voting_layer import Voting
from concept_learning_matching.displacement_layer import Displacement
from concept_learning_matching.utils.feature_align import feature_align
from concept_learning_matching.PCA.gconv import Siamese_Gconv
from concept_learning_matching.PCA.siamese_reducer import Siamese_Reducer
from concept_learning_matching.PCA.affinity_layer import Affinity

from concept_learning_matching.utils.config import cfg

import concept_learning_matching.utils.backbone
CNN = eval('concept_learning_matching.utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON)
        self.voting_layer = Voting(alpha=20)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        self.reducer = nn.Linear(4096,500)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))

    def forward(self, emb_src, emb_tgt, A_src, A_tgt, ns_src=15, ns_tgt=15):
        emb_src = self.reducer(emb_src)
        emb_tgt = self.reducer(emb_tgt)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])
            # emb_src = gnn_layer(emb_src, edge_index_src)
            # print(f'{i}: {emb_src.min()}-{emb_src.max()}')
            # print('----------------------------')

            # emb_tgt = gnn_layer(emb_tgt, edge_index_tgt)
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb_src, emb_tgt)
            ns_src = torch.tensor([A_src.shape[0]]).repeat(1,1,1)
            s = self.voting_layer(s, ns_src)
            s = self.bi_stochastic(s, ns_src)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_src, torch.bmm(s.transpose(1, 2), emb_tgt)), dim=-1))
                emb_src = emb1_new
                emb_tgt = emb2_new


        return s, emb_src, emb_tgt
