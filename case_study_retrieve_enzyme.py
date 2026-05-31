import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ERCP_FixRETR, PreNormResidualFFN


class CandidateEnzymeDataset(Dataset):
    def __init__(self, seq_ids: List[str], gvp_dir: str, esm_data: Dict[str, torch.Tensor]):
        self.seq_ids = [str(x) for x in seq_ids]
        self.gvp_dir = gvp_dir
        self.esm_data = esm_data

    def __len__(self) -> int:
        return len(self.seq_ids)

    def __getitem__(self, idx: int):
        seq_id = self.seq_ids[idx]
        gvp_path = os.path.join(self.gvp_dir, f"{seq_id}.pt")
        if not os.path.exists(gvp_path):
            return None

        gvp_graph = torch.load(gvp_path, map_location="cpu", weights_only=False)
        esm_vec = self.esm_data.get(seq_id)
        if esm_vec is None:
            return None

        return {
            "seq_id": seq_id,
            "gvp_graph": gvp_graph,
            "esm_emb": torch.as_tensor(esm_vec, dtype=torch.float32),
        }


def enzyme_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    gvp_batch = Batch.from_data_list([x["gvp_graph"] for x in batch])
    esm_batch = torch.stack([x["esm_emb"] for x in batch], dim=0)
    seq_ids = [x["seq_id"] for x in batch]
    return gvp_batch, esm_batch, seq_ids


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format is not a state_dict.")

    has_module_prefix = all(str(k).startswith("module.") for k in ckpt.keys())
    if has_module_prefix:
        ckpt = {k[len("module.") :]: v for k, v in ckpt.items()}
    return ckpt


def count_blocks(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    idx_set = set()
    for k in state_dict:
        m = pattern.match(k)
        if m:
            idx_set.add(int(m.group(1)))
    return max(idx_set) + 1 if idx_set else 0


def infer_model_config(
    state_dict: Dict[str, torch.Tensor],
    default_esm_dim: int,
    default_rxn_dim: int,
    default_fusion_dim: int,
) -> Dict[str, int]:
    esm_dim = default_esm_dim
    rxn_dim = default_rxn_dim
    fusion_dim = default_fusion_dim

    if "esm_proj_in.weight" in state_dict:
        fusion_dim = int(state_dict["esm_proj_in.weight"].shape[0])
        esm_dim = int(state_dict["esm_proj_in.weight"].shape[1])
    if "rxn_proj_in.weight" in state_dict:
        fusion_dim = int(state_dict["rxn_proj_in.weight"].shape[0])
        rxn_dim = int(state_dict["rxn_proj_in.weight"].shape[1])

    gvp_layers = count_blocks(state_dict, "gvp_layers") or 6
    rxn_blocks = count_blocks(state_dict, "rxn_blocks") or 2

    return {
        "esm_dim": esm_dim,
        "rxn_dim": rxn_dim,
        "fusion_dim": fusion_dim,
        "gvp_layers": gvp_layers,
        "rxn_blocks": rxn_blocks,
    }


def adapt_model_to_checkpoint(model: ERCP_FixRETR, state_dict: Dict[str, torch.Tensor], dropout: float):
    rxn_block_count = count_blocks(state_dict, "rxn_blocks")
    if rxn_block_count and rxn_block_count != len(model.rxn_blocks):
        hidden_dim = model.rxn_proj_in.out_features
        model.rxn_blocks = nn.ModuleList(
            [PreNormResidualFFN(hidden_dim, dropout=dropout) for _ in range(rxn_block_count)]
        )


@torch.no_grad()
def encode_candidate_enzymes(
    model,
    candidate_seq_ids: List[str],
    gvp_dir: str,
    esm_data: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, List[str]]:
    ds = CandidateEnzymeDataset(candidate_seq_ids, gvp_dir, esm_data)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=enzyme_collate,
    )

    vec_parts = []
    valid_seq_ids = []
    for batch in tqdm(loader, desc="Encode candidate enzymes"):
        if batch is None:
            continue
        gvp_batch, esm_batch, seq_ids = batch
        vec = model.encode_enzyme(gvp_batch.to(device), esm_batch.to(device))
        vec_parts.append(vec.cpu())
        valid_seq_ids.extend(seq_ids)

    if not vec_parts:
        raise RuntimeError("No candidate enzyme was encoded. Check gvp_dir / esm_emb coverage.")

    return torch.cat(vec_parts, dim=0), valid_seq_ids


@torch.no_grad()
def score_query_reaction_to_enzymes(
    model,
    query_rxn_vec: torch.Tensor,
    enzyme_vecs: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = query_rxn_vec.to(device)
    score_parts = []
    ph_parts = []
    temp_parts = []

    for s in range(0, enzyme_vecs.size(0), chunk_size):
        e = enzyme_vecs[s : s + chunk_size].to(device)
        n = e.size(0)
        r = q.expand(n, -1)
        fused = model.induced_fit(e, r)
        score_parts.append(model.predict_head(fused).squeeze(-1).cpu())
        ph_parts.append(model.predict_ph_head(fused).squeeze(-1).cpu())
        temp_parts.append(model.predict_temp_head(fused).squeeze(-1).cpu())

    scores = torch.cat(score_parts, dim=0).numpy()
    ph_raw = torch.cat(ph_parts, dim=0).numpy()
    temp_raw = torch.cat(temp_parts, dim=0).numpy()
    return scores, ph_raw, temp_raw


def infer_rxn_dim(rxn_data: Dict[str, torch.Tensor]) -> int:
    if not rxn_data:
        raise RuntimeError("Reaction embedding file is empty.")
    first_key = next(iter(rxn_data.keys()))
    first_vec = torch.as_tensor(rxn_data[first_key])
    if first_vec.ndim != 1:
        raise RuntimeError("Reaction embedding must be 1D vector.")
    return int(first_vec.shape[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Case study: retrieve enzymes for one reaction and predict pH/temp.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rxn_id", type=str, required=True)

    parser.add_argument("--data_csv", type=str, default="/data2/caiyueyi/ercp/data/case_study/case.csv")
    parser.add_argument(
        "--candidate_csv",
        type=str,
        default="/data2/caiyueyi/ercp/data/case_study/candidate_enzyme_uniref50.csv",
        # default="/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv",
    )
    parser.add_argument("--gvp_dir", type=str, default="/data2/caiyueyi/ercp/data/case_study/pt")
    # parser.add_argument("--gvp_dir", type=str, default="/data2/caiyueyi/dataset/pocket_dataset/processed_tensors/")
    parser.add_argument(
        "--esm_emb",
        type=str,
        default="/data2/caiyueyi/ercp/data/case_study/case_esm_sequence_embeddings.pt",
        # default="/data2/caiyueyi/dataset/esm_sequence_embeddings.pt"
    )
    parser.add_argument(
        "--rxn_emb",
        type=str,
        default="/data2/caiyueyi/ercp/data/case_study/case_rxn_unimol_embeddings.pt",
        # default="/data2/caiyueyi/ercp/data/case_study/case_rxn_rxnfp_embeddings.pt",
    )

    parser.add_argument("--esm_dim", type=int, default=1280)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--encode_batch_size", type=int, default=512)
    parser.add_argument("--score_chunk_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)

    # Optional de-normalization settings. Keep raw outputs too.
    parser.add_argument("--ph_mean", type=float, default=7.42)
    parser.add_argument("--ph_std", type=float, default=0.82)
    parser.add_argument("--temp_mean", type=float, default=35.62)
    parser.add_argument("--temp_std", type=float, default=11.80)

    parser.add_argument("--save_csv", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pair_df = pd.read_csv(args.data_csv)
    pair_df["seq_id"] = pair_df["seq_id"].astype(str)
    pair_df["rxn_id"] = pair_df["rxn_id"].astype(str)

    candidate_df = pd.read_csv(args.candidate_csv)
    candidate_df["seq_id"] = candidate_df["seq_id"].astype(str)
    candidate_seq_ids = candidate_df["seq_id"].dropna().astype(str).unique().tolist()
    if not candidate_seq_ids:
        raise RuntimeError("candidate_csv has no seq_id.")

    esm_data = torch.load(args.esm_emb, map_location="cpu", weights_only=False)
    rxn_data = torch.load(args.rxn_emb, map_location="cpu", weights_only=False)
    esm_data = {str(k): v for k, v in esm_data.items()}
    rxn_data = {str(k): v for k, v in rxn_data.items()}

    rxn_id = str(args.rxn_id)
    model = ERCP_FixRETR(
        esm_dim=1280,
        rxn_dim=512,
        fusion_dim=256,
        gvp_layers=6,
        dropout=args.dropout,
    )

    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device).eval()

    candidate_enzyme_vecs, valid_candidate_seq_ids = encode_candidate_enzymes(
        model=model,
        candidate_seq_ids=candidate_seq_ids,
        gvp_dir=args.gvp_dir,
        esm_data=esm_data,
        device=device,
        batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
    )

    query_rxn_emb = torch.as_tensor(rxn_data[rxn_id], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        query_rxn_vec = model.encode_reaction(query_rxn_emb).cpu()

    scores, ph_raw, temp_raw = score_query_reaction_to_enzymes(
        model=model,
        query_rxn_vec=query_rxn_vec,
        enzyme_vecs=candidate_enzyme_vecs,
        device=device,
        chunk_size=args.score_chunk_size,
    )

    positive_seq_set = set(
        pair_df.loc[pair_df["rxn_id"] == rxn_id, "seq_id"].astype(str).dropna().tolist()
    )

    sort_idx = np.argsort(-scores)
    rows = []
    for rank, i in enumerate(sort_idx, start=1):
        seq_id = valid_candidate_seq_ids[int(i)]
        rows.append(
            {
                "rank": rank,
                "seq_id": seq_id,
                "score": float(scores[i]),
                "pred_ph_raw": float(ph_raw[i]),
                "pred_temp_raw": float(temp_raw[i]),
                "pred_ph": float(ph_raw[i]) * float(args.ph_std) + float(args.ph_mean),
                "pred_temp": float(temp_raw[i]) * float(args.temp_std) + float(args.temp_mean),
                "is_positive_pair": bool(seq_id in positive_seq_set),
            }
        )

    out_df = pd.DataFrame(rows)

    print(f"Query reaction: {rxn_id}")
    print(f"Candidate enzymes scored: {len(out_df)}")
    print(f"Positive seq in case.csv for this reaction: {len(positive_seq_set)}")

    show_cols = [
        "rank",
        "seq_id",
        "score",
        "pred_ph",
        "pred_temp",
        "is_positive_pair",
    ]
    print("\nTop results:")
    print(out_df.head(args.top_k)[show_cols].to_string(index=False))

    if args.save_csv:
        save_csv = args.save_csv
    else:
        safe_rxn = re.sub(r"[^A-Za-z0-9_.-]+", "_", rxn_id)
        save_csv = f"/data2/caiyueyi/ercp/results/case_study/rxn_{safe_rxn}_enzyme_rank.csv"

    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    out_df.to_csv(save_csv, index=False)
    print(f"\nSaved: {save_csv}")


if __name__ == "__main__":
    main()
