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


class PerceptPairsSynDataset(InMemoryDataset):

    types = [
        'temp_ad.png', 'temp_al.png', 'temp_ar.png', 'temp_au.png',
        'temp_circle.png', 'temp_key.png', 'temp_d.png', 'goal.png']

    def __init__(self, root, name, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        super(PerceptPairsSynDataset, self).__init__(root, transform, pre_transform,
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
        return osp.join(f'{os.getcwd()}', 'percepts')

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['training_percepts.pt']

    @property
    def processed_dir(self):
        dir_to_save = os.path.join(os.getcwd(), 'caches_ds_synthetic')
        return dir_to_save

    def process(self):
        all_data = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for file in self.raw_file_names[:30]:
            with open(f'{self.raw_dir}/{file}', "rb") as output_file:
                pickled_episode_buffer = pickle.load(output_file)
                episodes_data = pickled_episode_buffer.sample(batch_size=256)
                nodes, action, reward, _, _, _, nr_templates, nr_proposals, template_labels = episodes_data
            for idx in range(len(nodes)-1):

                nodes_g1, action_g1, reward_g1, nr_templates_g1, nr_proposals_g1, template_labels_g1 = nodes[idx], action[idx], \
                                                                                                       reward[idx], nr_templates[idx], \
                                                                                                       nr_proposals[idx], template_labels[idx]
                nodes_g2, action_g2, reward_g2, nr_templates_g2, nr_proposals_g2, template_labels_g2 = nodes[idx+1], action[idx+1], \
                                                                                                       reward[idx+1], nr_templates[idx+1], \
                                                                                                       nr_proposals[idx+1], template_labels[idx+1]

                if (nr_templates_g1 + nr_templates_g1) > 0 and (nr_templates_g2 + nr_templates_g2) > 0:
                    full_nodes = nodes_g1[:, :(nr_proposals_g1 + nr_templates_g1), :]
                    only_template_nodes = nodes_g1[:, nr_proposals_g1:(nr_proposals_g1 + nr_templates_g1), :]
                    synthetic_g1 = []
                    synthetic_g2 = []
                    max_nr_of_rand = min((15 - (nr_proposals_g1 + nr_templates_g1)),
                                         (15 - (nr_proposals_g2 + nr_templates_g2)))

                    if (nr_proposals_g1 + nr_templates_g1) < 15 and (nr_proposals_g2 + nr_templates_g2) < 15:

                        if max_nr_of_rand > 8:
                            rand_g1 = torch.rand((8, 4096)).unsqueeze(0).to(device)
                            rand_g2 = torch.rand((8, 4096)).unsqueeze(0).to(device)
                            nodes_g1[:, (nr_proposals_g1 + nr_templates_g1) + 1:(nr_proposals_g1 + nr_templates_g1) + 9,
                            :] = rand_g1
                            nodes_g2[:, (nr_proposals_g2 + nr_templates_g2) + 1:(nr_proposals_g2 + nr_templates_g2) + 9,
                            :] = rand_g2
                            synthetic_g1 = [(nr_proposals_g1 + nr_templates_g1) + 1,
                                            (nr_proposals_g1 + nr_templates_g1) + 2,
                                            (nr_proposals_g1 + nr_templates_g1) + 3,
                                            (nr_proposals_g1 + nr_templates_g1) + 4,
                                            (nr_proposals_g1 + nr_templates_g1) + 5,
                                            (nr_proposals_g1 + nr_templates_g1) + 6,
                                            (nr_proposals_g1 + nr_templates_g1) + 7,
                                            (nr_proposals_g1 + nr_templates_g1) + 8,
                                            ]
                            synthetic_g2 = [(nr_proposals_g2 + nr_templates_g2) + 1,
                                            (nr_proposals_g2 + nr_templates_g2) + 2,
                                            (nr_proposals_g2 + nr_templates_g2) + 3,
                                            (nr_proposals_g2 + nr_templates_g2) + 4,
                                            (nr_proposals_g2 + nr_templates_g2) + 5,
                                            (nr_proposals_g2 + nr_templates_g2) + 6,
                                            (nr_proposals_g2 + nr_templates_g2) + 7,
                                            (nr_proposals_g2 + nr_templates_g2) + 8,
                                            ]

                    x1 = nodes_g1
                    x2 = nodes_g2
                    adj_1 = self.init_adj_matrix(x1)
                    adj_2 = self.init_adj_matrix(x2)
                    train_y, all_pos = self.process_permutation_matrix(template_labels_g1, template_labels_g2, nodes_g1,
                                                                       nodes_g2)
                    if max_nr_of_rand > 8:
                        for (i, j) in zip(synthetic_g1, synthetic_g2):
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
                                    n1_gt=torch.tensor([nr_proposals_g1 + nr_templates_g1]),
                                    n2_gt=torch.tensor([nr_proposals_g2 + nr_templates_g2]))
                        all_data.append(data)

        torch.save(self.collate(all_data), self.processed_paths[0])

    def init_adj_matrix(self, x):
        adj_matrix = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32)
        for i in range(x.shape[1]-1):
            j = i + 1
            xi = x[:, i, :]
            xj = x[:, j, :]
            adj_matrix[i][j] = torch.exp(torch.div(torch.cdist(xi, xj, p=2.0), 10)).item()
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

    def process_permutation_matrix(self, labels_g1, labels_g2, nodes_g1, nodes_g2):
        nr_nodes_1 = nodes_g1.shape[1]
        nr_nodes_2 = nodes_g2.shape[1]
        arr_1 = list(labels_g1)
        arr_2 = list(labels_g2)
        arr = arr_1 if len(arr_1) > len(arr_2) else arr_2
        smaller_arr = arr_2 if len(arr_1) > len(arr_2) else arr_1
        positions = []
        rev_positions = []
        all_pos = []
        for idx, el in enumerate(arr):
            if el in smaller_arr:
                idx_sec = smaller_arr.index(el)
                smaller_arr[idx_sec] = 'X'
                positions.append([idx, idx_sec])
                rev_positions.append([idx_sec, idx])
                all_pos.append([idx, idx_sec])
                if idx_sec != idx:
                    all_pos.append([idx_sec, idx])

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
    ds = PerceptPairsSynDataset( os.path.join(os.getcwd(), 'caches_ds_synthetic'),'percepts_ds')
    print(len(ds))
    return

if __name__ == '__main__':
    main()