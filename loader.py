import os
import json
import zipfile
import requests
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
import dgl


def split_train_val_test(
    total_size,
    ratio_train_val_test=None,
    n_train_val_test=None,
    keep_order=False,
    split_seed=7,
):
    if ratio_train_val_test:
        assert len(ratio_train_val_test) == 3
        ratio_train, ratio_val, ratio_test = ratio_train_val_test
        n_train = int(ratio_train * total_size)
        n_val = int(ratio_val * total_size) if ratio_val else None
        n_test = int(ratio_test * total_size) if ratio_test else None
    elif n_train_val_test:
        assert len(n_train_val_test) == 3
        n_train, n_val, n_test = n_train_val_test
    else:
        raise ValueError("Please Specify the dataset division.")

    ids = list(np.arange(total_size))
    if not keep_order:
        random.seed(split_seed)
        random.shuffle(ids)

    train_idx = ids[:n_train]
    val_idx = ids[n_train: n_train+n_val] if n_val else None
    test_idx = ids[-n_test:] if n_test else None
    return train_idx, val_idx, test_idx


def get_dataset(
    data_path,
    transforms=None,
    ratio_train_val_test=None,
    n_train_val_test=None,
    dihedral_graph=False
):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)

    train_idx, val_idx, test_idx = split_train_val_test(
                                        total_size=len(data),
                                        ratio_train_val_test=ratio_train_val_test,
                                        n_train_val_test=n_train_val_test)
    
    train_data = [data[idx] for idx in train_idx]
    val_data = [data[idx] for idx in val_idx] if val_idx else None
    test_data = [data[idx] for idx in test_idx] if val_idx else None

    train_transform = val_transform = test_transform = transforms

    train_dataset = CrystalGraphDataset(data=train_data, transforms=train_transform, dihedral_graph=dihedral_graph)
    val_dataset = CrystalGraphDataset(data=val_data, transforms=val_transform, dihedral_graph=dihedral_graph) if val_data else None
    test_dataset = CrystalGraphDataset(data=test_data, transforms=test_transform, dihedral_graph=dihedral_graph) if test_data else None

    return train_dataset, val_dataset, test_dataset


class CrystalGraphDataset(Dataset):
    def __init__(self, data, transforms, cutoff=5, max_neighbors=12, dihedral_graph=False):
        self.data = data
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.transforms = transforms
        self.dihedral_graph = dihedral_graph

    def __getitem__(self, index):
        info = self.data[index]
        crystal = dict()
        crystal['info'] = info
        crystal['structure'] = Atoms.from_dict(info['atoms'])
        crystal['graph'] = self.build_graph(crystal['structure'])
        crystal['line_graph'] = self.build_line_graph(crystal['graph'])
        if self.dihedral_graph:
            crystal['dihedral_graph'] = self.build_line_graph(crystal['line_graph'])
        inputs, targets = self.transforms(crystal)
        return inputs, targets

    def __len__(self):
        return len(self.data)
    
    def build_graph(self, atoms):
        edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=True,
            )
        u, v, r = build_undirected_edgedata(atoms, edges)
        g = dgl.graph((u, v))
        g.edata['offset'] = r
        return g

    def build_line_graph(self, g):
        lg = g.line_graph(shared=True)
        return lg

    #整合成小批量
    def collect(self):
        def collect_graphs(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch([g[0] for g in graphs])
            batched_line_graph = dgl.batch([g[1] for g in graphs])
            return [batched_graph, batched_line_graph], torch.stack(labels)

        def collect_dihedral_graph(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch([g[0] for g in graphs])
            batched_line_graph = dgl.batch([g[1] for g in graphs])
            batched_dihedral_graph = dgl.batch([g[2] for g in graphs])
            return [batched_graph, batched_line_graph, batched_dihedral_graph], torch.stack(labels)
        
        if self.dihedral_graph:
            return collect_dihedral_graph
        return collect_graphs



