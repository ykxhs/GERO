import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_chain_edges(num_nodes: int) -> torch.Tensor:
    src = np.arange(num_nodes - 1, dtype=np.int64)
    dst = src + 1
    edge_index = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def build_graph(num_nodes: int, rng: np.random.Generator) -> Data:
    seq = torch.tensor(rng.integers(0, 20, size=(num_nodes,), dtype=np.int64))
    node_v = torch.tensor(rng.normal(size=(num_nodes, 3, 3)), dtype=torch.float32)
    edge_index = build_chain_edges(num_nodes)
    edge_attr = torch.tensor(rng.normal(size=(edge_index.size(1), 1, 3)), dtype=torch.float32)
    return Data(seq=seq, node_v=node_v, edge_index=edge_index, edge_attr=edge_attr)


def save_embeddings(path: str, emb_map: Dict[str, torch.Tensor]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb_map, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--esm_dim", type=int, default=64)
    parser.add_argument("--rxn_model", type=str, default="rxnfp", choices=["ChemBERTa", "rxnfp", "unimol"])
    parser.add_argument("--num_nodes", type=int, default=6)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    seq_ids = ["S1", "S2", "S3", "S4"]
    rxn_ids = ["R1", "R2", "R3", "R4"]

    rows: List[Dict[str, object]] = []
    for i, (sid, rid) in enumerate(zip(seq_ids, rxn_ids)):
        rows.append(
            {
                "seq_id": sid,
                "rxn_id": rid,
                "seq": "ACDEFGHIK"[: 5 + (i % 4)],
                "rxn_smiles": "CCO>>CC=O" if i % 2 == 0 else "CCN>>CC[NH2+]",
                "ph_opt": float(6.5 + i * 0.5),
                "temp_opt": float(25 + i * 5),
                "enzyme_set": "train" if i < 2 else "test",
                "reaction_set": "train" if i < 2 else "test",
                "af_db": f"AF-{sid}-F1-model_v4.pdb",
            }
        )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "enzyme_reaction_data_splits.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    esm_emb = {}
    for sid in seq_ids:
        esm_emb[sid] = torch.tensor(rng.normal(size=(args.esm_dim,)), dtype=torch.float32)
    save_embeddings(os.path.join(out_dir, "esm_sequence_embeddings.pt"), esm_emb)

    if args.rxn_model == "ChemBERTa":
        rxn_dim = 768
        rxn_file = "rxn_ChemBERTa_embeddings.pt"
    elif args.rxn_model == "rxnfp":
        rxn_dim = 256
        rxn_file = "rxn_rxnfp_embeddings.pt"
    else:
        rxn_dim = 512
        rxn_file = "rxn_unimol_embeddings.pt"

    rxn_emb = {}
    for rid in rxn_ids:
        rxn_emb[rid] = torch.tensor(rng.normal(size=(rxn_dim,)), dtype=torch.float32)
    save_embeddings(os.path.join(out_dir, rxn_file), rxn_emb)

    gvp_dir = os.path.join(out_dir, "pocket_dataset", "processed_tensors")
    os.makedirs(gvp_dir, exist_ok=True)
    for sid in seq_ids:
        graph = build_graph(args.num_nodes, rng)
        torch.save(graph, os.path.join(gvp_dir, f"{sid}.pt"))

    print("Demo data written to:")
    print(f"  {out_dir}")
    print(f"  {csv_path}")
    print(f"  {os.path.join(out_dir, 'esm_sequence_embeddings.pt')}")
    print(f"  {os.path.join(out_dir, rxn_file)}")
    print(f"  {gvp_dir}")


if __name__ == "__main__":
    main()
