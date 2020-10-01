from concept_learning_matching.concept_ds import PerceptPairsDataset
# from concept_learning_matching.concept_ds_synthetic_matches import PerceptPairsSynDataset
from concept_learning_matching.concept_ds_mixmatch import PerceptPairsDatasetMix

from PIL import Image
import matplotlib.patches as patches

import networkx as nx
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from concept_learning_matching.utils.config import cfg
from concept_learning_matching.PCA.model import Net
import torch.nn.functional as F
import torch

import os
ds = PerceptPairsDataset(os.path.join(os.getcwd(), 'caches_ds_tempc'), 'correct_percepts')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_gr = Net()
model_gr.load_state_dict(
    torch.load('new_adj_pca_gm_frames_temp_2gnn_200ep_fix_bug_1.0e-3_500_14.pth', map_location=device)
)

model_gr.to(device)
model_gr.eval()


def run_model_pca_gm(model, inputs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_gr = Net()
    model_gr.load_state_dict(
        torch.load('new_adj_pca_gm_frames_temp_2gnn_200ep_fix_bug_1.0e-3_500_14.pth', map_location=device)
    )
    model_gr.to(device)
    model_gr.eval()
    emb_src = inputs.x1.to(device)
    emb_tgt = inputs.x2.to(device)
    perm_mat = inputs.train_y.unsqueeze(0).to(device)
    a_src = inputs.edge_attr1.to(device)
    a_tgt = inputs.edge_attr2.to(device)
    pos_matches = inputs.all_pos
    n1_gt = inputs.n1_gt.to(device)
    n2_gt = inputs.n2_gt.to(device)
    s_pred, new_emb_src, new_emb_tgt = model(emb_src, emb_tgt, a_src, a_tgt)


    return new_emb_src, new_emb_tgt

def draw_result(graph_to_draw, data):
    pos = [tuple(pos) for pos in data.pos.numpy()]
    edgewidth_g = [ float(d['weight']) for (u,v,d) in graph_to_draw.edges(data=True) if float(d['weight'])]
    with open('{}.png'.format(data.name[0]), 'rb') as f:
        plt.figure(1)
        img = Image.open(f).convert('RGB')
        img = img.resize((256, 256))
        plt.imshow(img)
        nx.draw_networkx(graph_to_draw,pos,width=edgewidth_g, edge_color='r')
    return

run_model(model_gr, ds[0])