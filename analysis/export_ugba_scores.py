#!/usr/bin/env python
"""Export the original UGBA cluster-degree score for Phase-2 correlation."""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn_extra import cluster
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.construct import model_construct
from utils import get_split, subgraph


def minmax(values):
    span = values.max() - values.min()
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=10)
    args_cli = parser.parse_args()

    args = SimpleNamespace(
        dataset="Cora", seed=args_cli.seed, hidden=32, dropout=0.5,
        train_lr=0.01, weight_decay=5e-4, epochs=200, target_class=0,
        dis_weight=1.0,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(
        root="./data/", name="Cora",
        transform=T.Compose([T.NormalizeFeatures()]),
    )
    data = dataset[0].to(device)
    data, idx_train, idx_val, _, _ = get_split(args, data, device)
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, _ = subgraph(
        torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False
    )
    unlabeled_idx = (
        torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)
    ).nonzero().flatten()

    encoder = model_construct(args, "GCN_Encoder", data, device).to(device)
    encoder.fit(
        data.x, train_edge_index, None, data.y, idx_train, idx_val,
        train_iters=args.epochs, verbose=False,
    )
    seen = torch.concat([idx_train, unlabeled_idx])
    embedding = encoder.get_h(data.x, train_edge_index, None).detach().cpu().numpy()
    nclass = int(data.y.max().item() + 1)
    kmedoids = cluster.KMedoids(n_clusters=nclass, method="pam")
    kmedoids.fit(embedding[seen.detach().cpu().numpy()])
    pred = kmedoids.predict(embedding)

    nodes = unlabeled_idx.detach().cpu().numpy().astype(np.int64)
    nodes = nodes[pred[nodes] != args.target_class]
    distances = np.linalg.norm(embedding - kmedoids.cluster_centers_[pred], axis=1)
    degrees = (
        degree(train_edge_index[0], num_nodes=data.num_nodes)
        + degree(train_edge_index[1], num_nodes=data.num_nodes)
    ).detach().cpu().numpy()
    normalized_distance = minmax(distances[nodes])
    normalized_degree = minmax(degrees[nodes])
    scores = normalized_distance + args.dis_weight * normalized_degree

    order = np.argsort(scores)
    with open(args_cli.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "node_id", "ugba_cluster_distance", "ugba_degree",
            "ugba_cluster_degree_score",
        ])
        writer.writeheader()
        for pos in order:
            writer.writerow({
                "node_id": int(nodes[pos]),
                "ugba_cluster_distance": float(normalized_distance[pos]),
                "ugba_degree": float(normalized_degree[pos]),
                "ugba_cluster_degree_score": float(scores[pos]),
            })
    print("wrote {} scores to {}".format(len(nodes), args_cli.output))
    print("bottom-40 node IDs: {}".format(nodes[order[:40]].tolist()))


if __name__ == "__main__":
    main()
