import os.path as osp
import glob
import os
import numpy as np
import shapely
import pickle
from matplotlib.path import Path


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


class PerceptPairsDatasetMix(InMemoryDataset):

    types = [
        'temp_ad.png', 'temp_al.png', 'temp_ar.png', 'temp_au.png',
        'temp_circle.png', 'temp_key.png', 'temp_d.png', 'goal.png']

    def __init__(self, root, name, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        super(PerceptPairsDatasetMix, self).__init__(root, transform, pre_transform,
                                             pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.dir = f'{os.getcwd()}'

        self.data, self.slices = torch.load(path)
        # path = osp.join(self.processed_dir, '{}_ged.pt'.format(self.name))
        # self.ged = torch.load(path)
        # path = osp.join(self.processed_dir, '{}_norm_ged.pt'.format(self.name))
        # self.norm_ged = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(f'{os.getcwd()}', 'correct_percepts2')

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['training_percepts.pt']

    @property
    def processed_dir(self):
        dir_to_save = os.path.join(os.getcwd(), 'caches_ds_mixtc2')
        return dir_to_save

    def process(self):
        all_data = []
        for file in self.raw_file_names[:30]:
            with open(f'{self.raw_dir}/{file}', "rb") as output_file:
                pickled_episode_buffer = pickle.load(output_file)
                print(file)


                episodes_data = pickled_episode_buffer.sample(batch_size=len(pickled_episode_buffer))
                nodes, action, reward, _, _, _, nr_templates, nr_proposals, template_labels, masks_roi, masks_temp = episodes_data
            for idx in range(len(nodes)-1):
                nodes_g1, action_g1, reward_g1, nr_templates_g1, nr_proposals_g1, template_labels_g1, masks_roi_g1, masks_temp_g1 \
                    = nodes[idx], action[idx], \
                      reward[idx], nr_templates[idx], \
                      nr_proposals[idx], template_labels[idx], \
                      masks_roi[idx], masks_temp[idx]

                nodes_g2, action_g2, reward_g2, nr_templates_g2, nr_proposals_g2, template_labels_g2, masks_roi_g2, masks_temp_g2 \
                    = nodes[idx+1], action[idx+1], \
                      reward[idx+1], nr_templates[idx+1], \
                      nr_proposals[idx+1], template_labels[idx+1], \
                      masks_roi[idx+1], masks_temp[idx+1]
                matches_temp_roi = self.find_coords_matches(masks_temp_g1, nr_proposals_g1.item(), masks_roi_g2)
                full_nodes = nodes_g1[:, :(nr_proposals_g1+nr_templates_g1), :]
                only_template_nodes = nodes_g1[:, nr_proposals_g1:(nr_proposals_g1+nr_templates_g1), :]
                x1 = nodes_g1
                x2 = nodes_g2
                adj_1 = self.init_adj_matrix(x1)
                adj_2 = self.init_adj_matrix(x2)
                train_y,all_pos = self.process_permutation_matrix(template_labels_g1, template_labels_g2, nodes_g1,
                                                                  nodes_g2, nr_proposals_g1, nr_proposals_g2)
                if len(matches_temp_roi) > 0:
                    for (i, j) in matches_temp_roi:
                        train_y[i, j] = 1
                        all_pos.append([i, j])
                        if i != j:
                            train_y[j, i] = 1
                            all_pos.append([j, i])
                edge_index1, edge_index2 = self.process_graph(nodes_g1, nodes_g2)
                if len(all_pos) != 0:
                    data = Data(x1=x1, edge_index1=edge_index1, x2=x2,
                                edge_index2=edge_index2,
                                edge_attr1=adj_1, edge_attr2=adj_2, train_y=train_y,
                                all_pos=all_pos,
                                n1_gt=torch.tensor([nr_proposals_g1+nr_templates_g1]),
                                n2_gt=torch.tensor([nr_proposals_g2+nr_templates_g2]))
                    all_data.append(data)
                    del masks_roi_g1
                    del masks_roi_g2
                    del masks_temp_g1
                    del masks_temp_g2
        torch.save(self.collate(all_data), self.processed_paths[0])

    def init_adj_matrix(self, x):
        adj_matrix = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32)
        for i in range(x.shape[1] - 1):
            j = i + 1
            xi = x[:, i, :]
            xj = x[:, j, :]
            adj_matrix[i][j] = torch.div(torch.cdist(xi, xj, p=2.0), 10).item()
        return adj_matrix

    def find_coords_matches(self, masks_temp_g1,rois_g1, masks_roi_g2):
        matches = []
        for idx_i, i in enumerate(masks_temp_g1):
            for idx_j, j in enumerate(masks_roi_g2):
                find_overlaps = torch.sum((i * j) == True)
                if find_overlaps.item() > 0:
                    matches.append((idx_i+rois_g1,idx_j))
        return matches



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
                positions.append([idx+arr_start, idx_sec+smaller_arr_start])
                rev_positions.append([idx_sec+smaller_arr_start, idx+arr_start])
                all_pos.append([idx+arr_start, idx_sec+smaller_arr_start])
                if idx_sec != idx:
                    all_pos.append([idx_sec+smaller_arr_start, idx+arr_start])

        permutation_matrix = torch.zeros((nr_nodes_1, nr_nodes_2), dtype=torch.int64)

        positions = np.array(positions)
        rev_positions = np.array(rev_positions)

        if len(positions) > 0:
            permutation_matrix[positions[:, 0], positions[:, 1]] = 1
            permutation_matrix[rev_positions[:, 0], rev_positions[:, 1]] = 1


        return permutation_matrix, all_pos

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pair)



def main():
    ds = PerceptPairsDatasetMix(os.path.join(os.getcwd(), 'caches_ds_mixtc2'), 'correct_percepts2')
    print(len(ds))
    return

if __name__ == '__main__':
    main()