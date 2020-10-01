import os.path as osp
import glob
import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip, extract_tar)
from torch_geometric.utils import to_undirected

import os
import os.path as osp
import shutil

import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index


class GraphDataUtils:

    types = [
        'temp_ad.png', 'temp_al.png', 'temp_ar.png', 'temp_au.png',
        'temp_circle.png', 'temp_key.png', 'temp_d.png', 'goal.png']

    def __init__(self):
        self.dir = f'{os.getcwd()}'


    def init_adj_matrix(self, x):
        adj_matrix = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32)
        for i in range(x.shape[1]-1):
            j = i + 1
            xi = x[:, i, :]
            xj = x[:, j, :]
            adj_matrix[i][j] = torch.div(torch.cdist(xi, xj, p=2.0), 10).item()
        return adj_matrix

    def generate_fully_connected_edge_index(self, nr_nodes):
        res = [[], []]
        for el in range(nr_nodes):
            first_lst = list(range(el + 1, nr_nodes))
            second_list = list(np.full(len(first_lst), el))
            res[0] = res[0] + second_list
            res[1] = res[1] + first_lst

        edge_index = torch.tensor(res, dtype=torch.long)
        return edge_index

    def process_graph(self, nodes_g1, nodes_g2):
        nr_nodes_1 = nodes_g1.shape[1]
        nr_nodes_2 = nodes_g2.shape[1]
        edge_index_1 = self.generate_fully_connected_edge_index(nr_nodes_1)
        edge_index_2 = self.generate_fully_connected_edge_index(nr_nodes_2)
        return edge_index_1, edge_index_2

    def process_permutation_matrix(self, labels_g1, labels_g2, nodes_g1, nodes_g2, nr_proposals_g1, nr_proposals_g2):
        nr_nodes_1 = nodes_g1.shape[1]
        nr_nodes_2 = nodes_g2.shape[1]
        arr_1 = list(labels_g1)
        arr_2 = list(labels_g2)
        arr = arr_1 if len(arr_1) > len(arr_2) else arr_2
        arr_start = nr_proposals_g1 if len(arr_1) > len(arr_2) else nr_proposals_g2
        smaller_arr = arr_2 if len(arr_1) > len(arr_2) else arr_1
        smaller_arr_start = nr_proposals_g2 if len(arr_1) > len(arr_2) else nr_proposals_g1
        positions = []
        rev_positions = []
        all_pos = []
        for idx, el in enumerate(arr):
            if el in smaller_arr:
                idx_sec = smaller_arr.index(el)
                smaller_arr[idx_sec] = 'X'
                positions.append([idx + arr_start, idx_sec + smaller_arr_start])
                rev_positions.append([idx_sec + smaller_arr_start, idx + arr_start])
                all_pos.append([idx + arr_start, idx_sec + smaller_arr_start])
                if idx_sec != idx:
                    all_pos.append([idx_sec + smaller_arr_start, idx + arr_start])

        permutation_matrix = torch.zeros((nr_nodes_1, nr_nodes_2), dtype=torch.int64)

        positions = np.array(positions)
        rev_positions = np.array(rev_positions)

        if len(positions) > 0:
            permutation_matrix[positions[:, 0], positions[:, 1]] = 1
            permutation_matrix[rev_positions[:, 0], rev_positions[:, 1]] = 1

        return permutation_matrix, all_pos

    def find_coords_matches(self, masks_temp_g1, rois_g1, masks_roi_g2):
        matches = []
        for idx_i, i in enumerate(masks_temp_g1):
            for idx_j, j in enumerate(masks_roi_g2):
                find_overlaps = torch.sum((i * j) == True)
                if find_overlaps.item() > 0:
                    matches.append((idx_i + rois_g1, idx_j))
        return matches

