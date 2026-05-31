import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
from src.model import ERCP_FixRETR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force deterministic kernels where possible for reproducible evaluation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sort_indices_desc_with_tie_break(scores: np.ndarray) -> np.ndarray:
    # Primary key: score descending; secondary key: candidate index ascending.
    return np.lexsort((np.arange(scores.shape[0]), -scores))

class UniqueEnzymeDataset(Dataset):
    def __init__(self, unique_ids, gvp_dir, esm_data):
        self.ids = [str(x) for x in unique_ids]
        self.gvp_dir = gvp_dir
        self.esm_data = esm_data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        gvp_path = os.path.join(self.gvp_dir, f"{seq_id}.pt")
        if not os.path.exists(gvp_path):
            return None

        try:
            gvp_graph = torch.load(gvp_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        esm_vec = self.esm_data.get(seq_id)
        if esm_vec is None:
            return None

        return {"gvp_graph": gvp_graph, "esm_emb": esm_vec, "id": seq_id}


class UniqueRxnDataset(Dataset):
    def __init__(self, unique_ids, rxn_data):
        self.ids = [str(x) for x in unique_ids]
        self.rxn_data = rxn_data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rid = self.ids[idx]
        rxn_vec = self.rxn_data.get(rid)
        if rxn_vec is None:
            return None
        return {"rxn_emb": rxn_vec, "id": rid}


def enzyme_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    gvp_batch = Batch.from_data_list([b["gvp_graph"] for b in batch])
    esm_embs = torch.stack([b["esm_emb"] for b in batch])
    ids = [b["id"] for b in batch]
    return gvp_batch, esm_embs, ids


def rxn_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    rxn_embs = torch.stack([b["rxn_emb"] for b in batch])
    ids = [b["id"] for b in batch]
    return rxn_embs, ids


def get_rxn_emb_path(data_path: str, rxn_model: str, user_path: str = None) -> Tuple[str, int]:
    if user_path:
        if rxn_model == "rxnfp":
            return user_path, 256
        if rxn_model == "ChemBERTa":
            return user_path, 768
        return user_path, 512

    if rxn_model == "ChemBERTa":
        return os.path.join(data_path, "rxn_ChemBERTa_embeddings.pt"), 768
    if rxn_model == "rxnfp":
        return os.path.join(data_path, "rxn_rxnfp_embeddings.pt"), 256
    return os.path.join(data_path, "rxn_unimol_embeddings.pt"), 512


def load_checkpoint_state_dict(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format is not a state_dict.")

    has_module_prefix = all(str(k).startswith("module.") for k in ckpt.keys())
    if has_module_prefix:
        ckpt = {k[len("module.") :]: v for k, v in ckpt.items()}
    return ckpt


def get_metrics(ranks: List[int], k_list=(1, 5, 10)):
    if len(ranks) == 0:
        return {"MRR": 0.0, **{f"Hit@{k}": 0.0 for k in k_list}}

    arr = np.array(ranks)
    return {
        "MRR": float(np.mean(1.0 / arr)),
        **{f"Hit@{k}": float(np.mean(arr <= k)) for k in k_list},
    }


def print_task_results(task_name: str, ranks: List[int], prop_hits: Dict[int, List[int]], k_list=(1, 5, 10)):
    metrics = get_metrics(ranks, k_list)
    print(f"\n[{task_name}]")
    print(f"Valid queries for ranking: {len(ranks)}")
    print(f"MRR: {metrics['MRR']:.4f}")
    for k in k_list:
        print(f"Hit@{k}: {metrics[f'Hit@{k}']:.4f}")

    for k in k_list:
        vals = prop_hits[k]
        if vals:
            print(f"PropHit@{k}: {float(np.mean(vals)):.4f} (n={len(vals)})")
        else:
            print(f"PropHit@{k}: N/A (n=0)")


def check_prop_valid(row) -> bool:
    if pd.isna(row["ph_opt"]) or pd.isna(row["temp_opt"]):
        return False
    return float(row["ph_opt"]) > 0 and float(row["temp_opt"]) > 0


@torch.no_grad()
def score_query_to_candidates(
    model,
    query_vec: torch.Tensor,
    candidate_vecs: torch.Tensor,
    query_is_enzyme: bool,
    device: torch.device,
    chunk_size: int,
):
    score_parts = []
    ph_parts = []
    temp_parts = []

    q = query_vec.to(device)
    for i in range(0, candidate_vecs.size(0), chunk_size):
        c = candidate_vecs[i : i + chunk_size].to(device)
        curr_c = c.size(0)

        if query_is_enzyme:
            e_flat = q.expand(curr_c, -1)
            r_flat = c
        else:
            e_flat = c
            r_flat = q.expand(curr_c, -1)

        feat = model.induced_fit(e_flat, r_flat)
        s = model.predict_head(feat).squeeze(-1)
        ph = model.predict_ph_head(feat).squeeze(-1)
        temp = model.predict_temp_head(feat).squeeze(-1)

        score_parts.append(s.cpu())
        ph_parts.append(ph.cpu())
        temp_parts.append(temp.cpu())

    scores = torch.cat(score_parts, dim=0).numpy()
    ph_norm = torch.cat(ph_parts, dim=0).numpy()
    temp_norm = torch.cat(temp_parts, dim=0).numpy()
    return scores, ph_norm, temp_norm


@torch.no_grad()
def compute_full_score_matrix(
    model,
    enz_vecs: torch.Tensor,
    rxn_vecs: torch.Tensor,
    device: torch.device,
    enzyme_batch_size: int = 256,
    rxn_chunk_size: int = 2048,
):
    num_enz = enz_vecs.size(0)
    num_rxn = rxn_vecs.size(0)
    score_matrix = np.zeros((num_enz, num_rxn), dtype=np.float32)

    for i in tqdm(range(0, num_enz, enzyme_batch_size), desc="Build Score Matrix"):
        e_batch = enz_vecs[i : i + enzyme_batch_size].to(device)
        curr_bs = e_batch.size(0)
        score_parts = []

        for j in range(0, num_rxn, rxn_chunk_size):
            r_chunk = rxn_vecs[j : j + rxn_chunk_size].to(device)
            curr_rxn = r_chunk.size(0)

            e_flat = e_batch.unsqueeze(1).expand(-1, curr_rxn, -1).reshape(-1, e_batch.size(1))
            r_flat = r_chunk.unsqueeze(0).expand(curr_bs, -1, -1).reshape(-1, r_chunk.size(1))

            fused = model.induced_fit(e_flat, r_flat)
            scores = model.predict_head(fused).squeeze(-1)
            score_parts.append(scores.view(curr_bs, curr_rxn).cpu())

        score_matrix[i : i + curr_bs] = torch.cat(score_parts, dim=1).numpy()

    return score_matrix


# The remainder of the file is unchanged from the working copy in the main project.
# It is included here to reproduce the evaluation pipeline used in the paper.


def evaluate_enzyme_to_reaction(
    model,
    raw_df,
    test_df,
    split_type,
    seq_id_to_idx,
    rxn_id_to_idx,
    enz_vecs,
    rxn_vecs,
    ph_mean,
    ph_std,
    temp_mean,
    temp_std,
    device,
    args,
    k_list=(1, 5, 10),
    cand_chunk_size=2048,
):
    print("\nEvaluating E-R (Enzyme -> Reaction)")

    candidate_df = raw_df if split_type == "enzyme" else test_df # reaction split: Only test reaction not seen in training (see enzyme -> new reaction )
    valid_rxn_ids = [rid for rid in candidate_df["rxn_id"].astype(str).unique() if rid in rxn_id_to_idx]
    cand_global_idx = np.array([rxn_id_to_idx[rid] for rid in valid_rxn_ids], dtype=np.int64)
    cand_vecs = rxn_vecs[cand_global_idx]

    print(f"E-R search space: {len(cand_global_idx)} reactions")

    cand_global_to_local = {int(g): i for i, g in enumerate(cand_global_idx.tolist())}

    er_ranks = []
    er_prop_hits = {k: [] for k in k_list}

    grouped = {sid: g for sid, g in test_df.groupby("seq_id")}

    for q_seq_id, group in tqdm(grouped.items(), desc="E-R"):
        q_seq_id = str(q_seq_id)
        if q_seq_id not in seq_id_to_idx:
            continue

        gt_global = []
        for rid in group["rxn_id"].astype(str).unique().tolist():
            if rid in rxn_id_to_idx:
                g = rxn_id_to_idx[rid]
                if g in cand_global_to_local:
                    gt_global.append(g)

        if not gt_global:
            continue

        q_idx = seq_id_to_idx[q_seq_id]
        query_vec = enz_vecs[q_idx : q_idx + 1]

        scores, ph_norm, temp_norm = score_query_to_candidates(
            model=model,
            query_vec=query_vec,
            candidate_vecs=cand_vecs,
            query_is_enzyme=True,
            device=device,
            chunk_size=cand_chunk_size,
        )

        sorted_indices = sort_indices_desc_with_tie_break(scores)
        rank_lookup = np.empty(len(scores), dtype=np.int64)
        rank_lookup[sorted_indices] = np.arange(1, len(scores) + 1)

        gt_local = [cand_global_to_local[g] for g in gt_global]
        best_rank = int(np.min(rank_lookup[gt_local]))
        er_ranks.append(best_rank)

        has_valid_prop = False
        hit_by_k = {k: False for k in k_list}

        for _, row in group.iterrows():
            if not check_prop_valid(row):
                continue

            rid = str(row["rxn_id"])
            g_idx = rxn_id_to_idx[rid]

            has_valid_prop = True
            local_idx = cand_global_to_local[g_idx]
            curr_rank = int(rank_lookup[local_idx])

            pred_ph = float(ph_norm[local_idx]) * ph_std + ph_mean
            pred_temp = float(temp_norm[local_idx]) * temp_std + temp_mean

            ph_ok = abs(pred_ph - float(row["ph_opt"])) <= args.ph_treshold
            temp_ok = abs(pred_temp - float(row["temp_opt"])) <= args.temp_treshold

            for k in k_list:
                if curr_rank <= k and ph_ok and temp_ok:
                    hit_by_k[k] = True

        if has_valid_prop:
            for k in k_list:
                er_prop_hits[k].append(1 if hit_by_k[k] else 0)

    print_task_results("Task E-R", er_ranks, er_prop_hits, k_list)

    metrics = get_metrics(er_ranks, k_list)
    for k in k_list:
        vals = er_prop_hits[k]
        metrics[f"propHit@{k}"] = float(np.mean(vals)) if vals else 0.0
        metrics[f"propValidCount@{k}"] = len(vals)
    return metrics


def evaluate_reaction_to_enzyme(
    model,
    raw_df,
    test_df,
    split_type,
    seq_id_to_idx,
    rxn_id_to_idx,
    enz_vecs,
    rxn_vecs,
    ph_mean,
    ph_std,
    temp_mean,
    temp_std,
    device,
    args,
    k_list=(1, 5, 10),
    cand_chunk_size=2048,
):
    print("\nEvaluating R-E (Reaction -> Enzyme)")

    candidate_df = test_df if split_type == "enzyme" else raw_df # enzyme split: Only test enzyme not seen in training (see reaction -> new enzyme )
    valid_seq_ids = [sid for sid in candidate_df["seq_id"].astype(str).unique() if sid in seq_id_to_idx]
    cand_global_idx = np.array([seq_id_to_idx[sid] for sid in valid_seq_ids], dtype=np.int64)
    cand_vecs = enz_vecs[cand_global_idx]

    print(f"R-E search space: {len(cand_global_idx)} enzymes")

    cand_global_to_local = {int(g): i for i, g in enumerate(cand_global_idx.tolist())}

    re_ranks = []
    re_prop_hits = {k: [] for k in k_list}

    grouped = {rid: g for rid, g in test_df.groupby("rxn_id")}

    for q_rxn_id, group in tqdm(grouped.items(), desc="R-E"):
        q_rxn_id = str(q_rxn_id)
        if q_rxn_id not in rxn_id_to_idx:
            continue

        gt_global = []
        for sid in group["seq_id"].astype(str).unique().tolist():
            if sid in seq_id_to_idx:
                g = seq_id_to_idx[sid]
                if g in cand_global_to_local:
                    gt_global.append(g)

        if not gt_global:
            continue

        q_idx = rxn_id_to_idx[q_rxn_id]
        query_vec = rxn_vecs[q_idx : q_idx + 1]

        scores, ph_norm, temp_norm = score_query_to_candidates(
            model=model,
            query_vec=query_vec,
            candidate_vecs=cand_vecs,
            query_is_enzyme=False,
            device=device,
            chunk_size=cand_chunk_size,
        )

        sorted_indices = sort_indices_desc_with_tie_break(scores)
        rank_lookup = np.empty(len(scores), dtype=np.int64)
        rank_lookup[sorted_indices] = np.arange(1, len(scores) + 1)

        gt_local = [cand_global_to_local[g] for g in gt_global]
        best_rank = int(np.min(rank_lookup[gt_local]))
        re_ranks.append(best_rank)

        has_valid_prop = False
        hit_by_k = {k: False for k in k_list}

        for _, row in group.iterrows():
            if not check_prop_valid(row):
                continue

            sid = str(row["seq_id"])

            g_idx = seq_id_to_idx[sid]

            has_valid_prop = True
            local_idx = cand_global_to_local[g_idx]
            curr_rank = int(rank_lookup[local_idx])

            pred_ph = float(ph_norm[local_idx]) * ph_std + ph_mean
            pred_temp = float(temp_norm[local_idx]) * temp_std + temp_mean

            ph_ok = abs(pred_ph - float(row["ph_opt"])) <= args.ph_treshold
            temp_ok = abs(pred_temp - float(row["temp_opt"])) <= args.temp_treshold

            for k in k_list:
                if curr_rank <= k and ph_ok and temp_ok:
                    hit_by_k[k] = True

        if has_valid_prop:
            for k in k_list:
                re_prop_hits[k].append(1 if hit_by_k[k] else 0)

    print_task_results("Task R-E", re_ranks, re_prop_hits, k_list)

    metrics = get_metrics(re_ranks, k_list)
    for k in k_list:
        vals = re_prop_hits[k]
        metrics[f"propHit@{k}"] = float(np.mean(vals)) if vals else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="enzyme", choices=["enzyme", "reaction"])
    parser.add_argument("--task", type=str, default="all", choices=["all", "er", "re"])

    parser.add_argument("--data_path", type=str, default="/data2/caiyueyi/dataset/")
    parser.add_argument("--data_csv", type=str, default="/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv")
    parser.add_argument("--gvp_dir", type=str, default="/data2/caiyueyi/dataset/pocket_dataset/processed_tensors")
    parser.add_argument("--esm_emb", type=str, default="/data2/caiyueyi/dataset/esm_sequence_embeddings.pt")
    parser.add_argument("--rxn_emb", type=str, default=None)
    parser.add_argument("--rxn_model", type=str, default="unimol", choices=["ChemBERTa", "rxnfp", "unimol"])

    parser.add_argument("--esm_dim", type=int, default=1280)
    parser.add_argument("--fusion_dim", type=int, default=256)

    parser.add_argument("--encode_batch_size", type=int, default=512)
    parser.add_argument("--score_chunk_size", type=int, default=2048)
    parser.add_argument("--matrix_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ph_treshold", type=float, default=0.5)
    parser.add_argument("--temp_treshold", type=float, default=5.0)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_matrix", type=str, default="/data2/caiyueyi/ercp/results/score_matrix")

    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_df = pd.read_csv(args.data_csv)
    raw_df["seq_id"] = raw_df["seq_id"].astype(str)
    raw_df["rxn_id"] = raw_df["rxn_id"].astype(str)

    train_df = raw_df[raw_df[f"{args.split}_set"] == "train"].copy()
    test_df = raw_df[raw_df[f"{args.split}_set"] == "test"].copy()

    ph_mean = float(train_df["ph_opt"].mean())
    ph_std = float(train_df["ph_opt"].std())
    temp_mean = float(train_df["temp_opt"].mean())
    temp_std = float(train_df["temp_opt"].std())

    rxn_emb_path, rxn_dim = get_rxn_emb_path(args.data_path, args.rxn_model, args.rxn_emb)
    print(f"Loading ESM embeddings: {args.esm_emb}")
    print(f"Loading reaction embeddings: {rxn_emb_path}")

    esm_data = torch.load(args.esm_emb, map_location="cpu", weights_only=False)
    rxn_data = torch.load(rxn_emb_path, map_location="cpu", weights_only=False)
    esm_data = {str(k): v for k, v in esm_data.items()}
    rxn_data = {str(k): v for k, v in rxn_data.items()}

    model = ERCP_FixRETR(
        esm_dim=args.esm_dim,
        rxn_dim=rxn_dim,
        fusion_dim=args.fusion_dim,
    )
    state_dict = load_checkpoint_state_dict(args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    unique_seq_ids = raw_df["seq_id"].astype(str).unique().tolist()
    unique_rxn_ids = raw_df["rxn_id"].astype(str).unique().tolist()

    enz_ds = UniqueEnzymeDataset(unique_seq_ids, args.gvp_dir, esm_data)
    rxn_ds = UniqueRxnDataset(unique_rxn_ids, rxn_data)

    enz_loader = DataLoader(
        enz_ds,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=enzyme_collate,
    )
    rxn_loader = DataLoader(
        rxn_ds,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=rxn_collate,
    )

    print("Encoding all candidate enzymes...")
    all_enz = []
    final_seq_ids = []
    with torch.no_grad():
        for batch in tqdm(enz_loader, desc="Encode Enzyme"):
            if batch is None:
                continue
            gvp_batch, esm_batch, ids = batch
            vec = model.encode_enzyme(gvp_batch.to(device), esm_batch.to(device))
            all_enz.append(vec.cpu())
            final_seq_ids.extend(ids)

    if not all_enz:
        raise RuntimeError("No enzyme vectors were encoded. Check gvp_dir / esm embeddings coverage.")
    enz_vecs = torch.cat(all_enz, dim=0)

    print("Encoding all candidate reactions...")
    all_rxn = []
    final_rxn_ids = []
    with torch.no_grad():
        for batch in tqdm(rxn_loader, desc="Encode Reaction"):
            if batch is None:
                continue
            rxn_batch, ids = batch
            vec = model.encode_reaction(rxn_batch.to(device))
            all_rxn.append(vec.cpu())
            final_rxn_ids.extend(ids)

    if not all_rxn:
        raise RuntimeError("No reaction vectors were encoded. Check reaction embedding coverage.")
    rxn_vecs = torch.cat(all_rxn, dim=0)

    seq_id_to_idx = {sid: i for i, sid in enumerate(final_seq_ids)}
    rxn_id_to_idx = {rid: i for i, rid in enumerate(final_rxn_ids)}

    print(
        f"Encoded matrix space: {len(final_seq_ids)} enzymes x {len(final_rxn_ids)} reactions "
        f"({len(seq_id_to_idx)} unique enzyme ids, {len(rxn_id_to_idx)} unique reaction ids)"
    )
    if args.save_matrix:
        print("Computing full score matrix for saving...")
        args.save_matrix = os.path.join(args.save_matrix, f"ERCP_{args.split}_split_score_matrix.npy")
        score_matrix = compute_full_score_matrix(
            model=model,
            enz_vecs=enz_vecs,
            rxn_vecs=rxn_vecs,
            device=device,
            enzyme_batch_size=args.matrix_batch_size,
            rxn_chunk_size=args.score_chunk_size,
        )

        save_dir = os.path.dirname(args.save_matrix)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        np.save(
            args.save_matrix,
            {
                "score_matrix": score_matrix,
                "row_ids": final_seq_ids,
                "col_ids": final_rxn_ids,
            },
        )
        print(f"Saved score matrix: {args.save_matrix}")
        print(f"Score matrix shape: {score_matrix.shape}")

    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "task": args.task,
        "metrics": {},
    }

    if args.task != "re":
        er_metrics = evaluate_enzyme_to_reaction(
            model=model,
            raw_df=raw_df,
            test_df=test_df,
            split_type=args.split,
            seq_id_to_idx=seq_id_to_idx,
            rxn_id_to_idx=rxn_id_to_idx,
            enz_vecs=enz_vecs,
            rxn_vecs=rxn_vecs,
            ph_mean=ph_mean,
            ph_std=ph_std,
            temp_mean=temp_mean,
            temp_std=temp_std,
            device=device,
            args=args,
            cand_chunk_size=args.score_chunk_size,
        )
        out["metrics"]["E-R"] = er_metrics

    if args.task != "er":
        re_metrics = evaluate_reaction_to_enzyme(
            model=model,
            raw_df=raw_df,
            test_df=test_df,
            split_type=args.split,
            seq_id_to_idx=seq_id_to_idx,
            rxn_id_to_idx=rxn_id_to_idx,
            enz_vecs=enz_vecs,
            rxn_vecs=rxn_vecs,
            ph_mean=ph_mean,
            ph_std=ph_std,
            temp_mean=temp_mean,
            temp_std=temp_std,
            device=device,
            args=args,
            cand_chunk_size=args.score_chunk_size,
        )
        out["metrics"]["R-E"] = re_metrics

    print("\n========== Summary ==========")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split type: {args.split}")
    print(f"Evaluated tasks: {args.task}")
    for task_name, metrics in out["metrics"].items():
        print(f"\n[{task_name}]")
        print(f"MRR: {metrics['MRR']:.4f}")
        for k in [1, 10]:
            print(f"Hit@{k}: {metrics[f'Hit@{k}']:.4f}")
            if f"propHit@{k}" in metrics:
                print(f"Hit@{k}-TR: {metrics[f'propHit@{k}']:.4f}")

    if args.save_json:
        save_dir = os.path.dirname(args.save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics json: {args.save_json}")


if __name__ == "__main__":
    main()
