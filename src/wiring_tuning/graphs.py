"""Connectome graph representation.

A connectome is a directed, weighted graph: neurons are nodes (with
cell-type features) and synapses are edges (with weights = synapse counts).
This module provides a lightweight container and subgraph sampling utilities
used by the benchmark. All operations are pure NumPy/PyTorch (no PyG needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ConnectomeGraph:
    """Directed, weighted connectome graph.

    Attributes:
    node_features: (N, F) float array of per-neuron features
    (e.g., one-hot cell type, layer depth).
    edge_index: (2, E) int64 array of (source, target) node indices.
    edge_weight: (E,) float array of synaptic weights (synapse counts).
    cell_types: (N,) int array of cell-type ids (0=exc, 1..K=inhibitory).
    node_ids: (N,) array of stable node identifiers (e.g., MICrONS cell IDs).
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    cell_types: np.ndarray
    node_ids: np.ndarray

    @property
    def num_nodes(self) -> int:
        """Num nodes.

        Returns:
        int: the nodes.
        """
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Num edges.

        Returns:
        int: the edges.
        """
        return self.edge_index.shape[1]

    def to_torch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, edge_index, edge_weight) as torch tensors."""
        x = torch.as_tensor(self.node_features, dtype=torch.float32)
        ei = torch.as_tensor(self.edge_index, dtype=torch.long)
        ew = torch.as_tensor(self.edge_weight, dtype=torch.float32)
        return x, ei, ew

    def in_out_degree(self) -> tuple[np.ndarray, np.ndarray]:
        """Weighted in/out degree per node."""
        n = self.num_nodes
        src, dst = self.edge_index
        out_deg = np.bincount(src, weights=self.edge_weight, minlength=n)
        in_deg = np.bincount(dst, weights=self.edge_weight, minlength=n)
        return in_deg, out_deg

    def degree_features(self) -> np.ndarray:
        """Deterministic hand-crafted connectivity features per node: [in_deg, out_deg, in+out, in-out, frac_inhibitory_input, frac_inhibitory_output]."""
        n = self.num_nodes
        src, dst, w = self.edge_index[0], self.edge_index[1], self.edge_weight
        in_deg, out_deg = self.in_out_degree()
        inh_mask = self.cell_types > 0
        in_inh = np.bincount(dst[inh_mask[src]], weights=w[inh_mask[src]], minlength=n)
        out_inh = np.bincount(src[inh_mask[dst]], weights=w[inh_mask[dst]], minlength=n)
        eps = 1e-9
        feats = np.stack(
            [
                in_deg,
                out_deg,
                in_deg + out_deg,
                in_deg - out_deg,
                in_inh / (in_deg + eps),
                out_inh / (out_deg + eps),
            ],
            axis=1,
        ).astype(np.float32)
        return feats


def sample_subgraph(graph: ConnectomeGraph, node_indices: np.ndarray) -> ConnectomeGraph:
    """Induced subgraph on the given node indices (edges fully inside kept)."""
    node_indices = np.asarray(node_indices)
    keep = np.zeros(graph.num_nodes, dtype=bool)
    keep[node_indices] = True
    ekeep = keep[graph.edge_index[0]] & keep[graph.edge_index[1]]
    remap = np.full(graph.num_nodes, -1, dtype=np.int64)
    remap[node_indices] = np.arange(len(node_indices))
    ei = remap[graph.edge_index[:, ekeep]]
    return ConnectomeGraph(
        node_features=graph.node_features[node_indices],
        edge_index=ei,
        edge_weight=graph.edge_weight[ekeep],
        cell_types=graph.cell_types[node_indices],
        node_ids=graph.node_ids[node_indices],
    )
