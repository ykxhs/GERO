import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from src.model import ERCP_FixRETR, PreNormResidualFFN


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

        return {
            "gvp_graph": gvp_graph,
            "esm_emb": torch.as_tensor(esm_vec, dtype=torch.float32),
            "id": seq_id,
        }


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

        return {
            "rxn_emb": torch.as_tensor(rxn_vec, dtype=torch.float32),
            "id": rid,
        }


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


def get_rxn_emb_path(data_path: str, rxn_model: str, user_path: Optional[str] = None) -> Tuple[str, int]:
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


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint format is not a state_dict.")

    has_module_prefix = all(str(key).startswith("module.") for key in checkpoint.keys())
    if has_module_prefix:
        checkpoint = {key[len("module.") :]: value for key, value in checkpoint.items()}
    return checkpoint


def count_block_count(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = set()
    for key in state_dict:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return max(indices) + 1 if indices else 0


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

    gvp_layers = count_block_count(state_dict, "gvp_layers") or 6
    rxn_blocks = count_block_count(state_dict, "rxn_blocks") or 2
    return {
        "esm_dim": esm_dim,
        "rxn_dim": rxn_dim,
        "fusion_dim": fusion_dim,
        "gvp_layers": gvp_layers,
        "rxn_blocks": rxn_blocks,
    }


def adapt_model_to_checkpoint(model: ERCP_FixRETR, state_dict: Dict[str, torch.Tensor], dropout: float):
    rxn_block_count = count_block_count(state_dict, "rxn_blocks")
    if rxn_block_count and rxn_block_count != len(model.rxn_blocks):
        hidden_dim = model.rxn_proj_in.out_features
        model.rxn_blocks = nn.ModuleList(
            [PreNormResidualFFN(hidden_dim, dropout=dropout) for _ in range(rxn_block_count)]
        )


def build_model(args, state_dict: Dict[str, torch.Tensor], rxn_dim: int, device: torch.device):
    inferred = infer_model_config(
        state_dict=state_dict,
        default_esm_dim=args.esm_dim,
        default_rxn_dim=rxn_dim,
        default_fusion_dim=args.fusion_dim,
    )

    model = ERCP_FixRETR(
        esm_dim=inferred["esm_dim"],
        rxn_dim=inferred["rxn_dim"],
        fusion_dim=inferred["fusion_dim"],
        gvp_layers=inferred["gvp_layers"],
    )
    adapt_model_to_checkpoint(model, state_dict, dropout=args.dropout)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, inferred


@torch.no_grad()
def encode_enzyme_dict(
    model,
    seq_ids: List[str],
    gvp_dir: str,
    esm_data: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    dataset = UniqueEnzymeDataset(seq_ids, gvp_dir, esm_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=enzyme_collate,
    )

    vec_parts = []
    valid_ids = []
    for batch in tqdm(loader, desc="Encode enzymes"):
        if batch is None:
            continue
        gvp_batch, esm_batch, ids = batch
        vecs = model.encode_enzyme(gvp_batch.to(device), esm_batch.to(device))
        vec_parts.append(vecs.cpu())
        valid_ids.extend(ids)

    if not vec_parts:
        raise RuntimeError("No valid enzyme embeddings were encoded.")

    all_vecs = torch.cat(vec_parts, dim=0)
    return {seq_id: all_vecs[idx] for idx, seq_id in enumerate(valid_ids)}


@torch.no_grad()
def encode_rxn_pool(
    model,
    rxn_ids: List[str],
    rxn_data: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    dataset = UniqueRxnDataset(rxn_ids, rxn_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rxn_collate,
    )

    vec_parts = []
    valid_ids = []
    for batch in tqdm(loader, desc="Encode reactions"):
        if batch is None:
            continue
        rxn_batch, ids = batch
        vecs = model.encode_reaction(rxn_batch.to(device))
        vec_parts.append(vecs.cpu())
        valid_ids.extend(ids)

    if not vec_parts:
        raise RuntimeError("No valid reaction embeddings were encoded.")

    all_vecs = torch.cat(vec_parts, dim=0)
    return all_vecs, valid_ids


@torch.no_grad()
def predict_pair(model, enzyme_vec: torch.Tensor, rxn_vec: torch.Tensor, device: torch.device) -> Tuple[float, float, float]:
    e = enzyme_vec.unsqueeze(0).to(device)
    r = rxn_vec.unsqueeze(0).to(device)
    fused = model.induced_fit(e, r)

    score = float(model.predict_head(fused).squeeze().cpu().item())
    pred_ph_norm = float(model.predict_ph_head(fused).squeeze().cpu().item())
    pred_temp_norm = float(model.predict_temp_head(fused).squeeze().cpu().item())
    return score, pred_ph_norm, pred_temp_norm


@torch.no_grad()
def predict_top1(
    model,
    enzyme_vec: torch.Tensor,
    rxn_pool_vecs: torch.Tensor,
    rxn_pool_ids: List[str],
    device: torch.device,
    chunk_size: int,
) -> Tuple[str, float, float, float]:
    e = enzyme_vec.to(device)

    best_score = None
    best_rxn = None
    best_ph = None
    best_temp = None

    for start in range(0, rxn_pool_vecs.size(0), chunk_size):
        c = rxn_pool_vecs[start : start + chunk_size].to(device)
        curr_n = c.size(0)

        e_flat = e.unsqueeze(0).expand(curr_n, -1)
        fused = model.induced_fit(e_flat, c)

        scores = model.predict_head(fused).squeeze(-1).cpu().numpy()
        ph_norm = model.predict_ph_head(fused).squeeze(-1).cpu().numpy()
        temp_norm = model.predict_temp_head(fused).squeeze(-1).cpu().numpy()

        local_idx = int(np.argmax(scores))
        local_score = float(scores[local_idx])
        if best_score is None or local_score > best_score:
            global_idx = start + local_idx
            best_score = local_score
            best_rxn = rxn_pool_ids[global_idx]
            best_ph = float(ph_norm[local_idx])
            best_temp = float(temp_norm[local_idx])

    if best_rxn is None:
        raise RuntimeError("Top1 retrieval failed because reaction pool is empty.")

    return best_rxn, best_score, best_ph, best_temp


@dataclass
class SampleItem:
    seq_id: str
    true_ph: Optional[float]
    true_temp: Optional[float]
    pos_rxn_ids: List[str]


def is_valid_value(value: float) -> bool:
    return pd.notna(value) and float(value) > 0


def build_enzyme_samples(test_df: pd.DataFrame) -> List[SampleItem]:
    samples: List[SampleItem] = []
    grouped = test_df.groupby("seq_id")

    for seq_id, group in grouped:
        ph_values = [float(x) for x in group["ph_opt"].tolist() if is_valid_value(x)]
        temp_values = [float(x) for x in group["temp_opt"].tolist() if is_valid_value(x)]

        if not ph_values and not temp_values:
            continue

        pos_rxn_ids = sorted(group["rxn_id"].astype(str).unique().tolist())
        if not pos_rxn_ids:
            continue

        samples.append(
            SampleItem(
                seq_id=str(seq_id),
                true_ph=float(np.mean(ph_values)) if ph_values else None,
                true_temp=float(np.mean(temp_values)) if temp_values else None,
                pos_rxn_ids=pos_rxn_ids,
            )
        )

    return samples


def calc_metrics(preds: List[float], targets: List[float]) -> Tuple[float, float]:
    if not preds:
        return float("nan"), float("nan")

    pred_arr = np.asarray(preds, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    err = pred_arr - target_arr

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return mae, rmse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--data_path", type=str, default="/data2/caiyueyi/dataset/")
    parser.add_argument("--data_csv", type=str, default="/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv")
    parser.add_argument("--gvp_dir", type=str, default="/data2/caiyueyi/dataset/pocket_dataset/processed_tensors")
    parser.add_argument("--esm_emb", type=str, default="/data2/caiyueyi/dataset/esm_sequence_embeddings.pt")
    parser.add_argument("--rxn_emb", type=str, default=None)
    parser.add_argument("--rxn_model", type=str, default="unimol", choices=["ChemBERTa", "rxnfp", "unimol"])

    parser.add_argument("--esm_dim", type=int, default=1280)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--encode_batch_size", type=int, default=512)
    parser.add_argument("--score_chunk_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_csv", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Evaluation split: enzyme")

    raw_df = pd.read_csv(args.data_csv)
    raw_df["seq_id"] = raw_df["seq_id"].astype(str)
    raw_df["rxn_id"] = raw_df["rxn_id"].astype(str)

    train_df = raw_df[raw_df["enzyme_set"] == "train"].copy()
    test_df = raw_df[raw_df["enzyme_set"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("enzyme split train/test rows are empty. Please check data_csv.")

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

    state_dict = load_checkpoint_state_dict(args.checkpoint)
    model, inferred = build_model(args, state_dict, rxn_dim, device)
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(esm_dim={inferred['esm_dim']}, rxn_dim={inferred['rxn_dim']}, "
        f"fusion_dim={inferred['fusion_dim']}, gvp_layers={inferred['gvp_layers']}, "
        f"rxn_blocks={inferred['rxn_blocks']})"
    )

    all_rxn_ids = sorted(raw_df["rxn_id"].astype(str).unique().tolist())
    rxn_pool_vecs, rxn_pool_ids = encode_rxn_pool(
        model=model,
        rxn_ids=all_rxn_ids,
        rxn_data=rxn_data,
        device=device,
        batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
    )
    rxn_vec_by_id = {rid: rxn_pool_vecs[i] for i, rid in enumerate(rxn_pool_ids)}

    samples = build_enzyme_samples(test_df)
    eval_seq_ids = sorted({sample.seq_id for sample in samples})
    enzyme_vec_by_id = encode_enzyme_dict(
        model=model,
        seq_ids=eval_seq_ids,
        gvp_dir=args.gvp_dir,
        esm_data=esm_data,
        device=device,
        batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
    )

    rng = random.Random(args.seed)

    rows = []
    skipped_no_enzyme = 0
    skipped_no_pos = 0

    for sample in tqdm(samples, desc="Evaluate enzyme samples"):
        enzyme_vec = enzyme_vec_by_id.get(sample.seq_id)
        if enzyme_vec is None:
            skipped_no_enzyme += 1
            continue

        valid_pos = [rid for rid in sample.pos_rxn_ids if rid in rxn_vec_by_id]
        if not valid_pos:
            skipped_no_pos += 1
            continue

        top1_rxn, _, top1_ph_norm, top1_temp_norm = predict_top1(
            model=model,
            enzyme_vec=enzyme_vec,
            rxn_pool_vecs=rxn_pool_vecs,
            rxn_pool_ids=rxn_pool_ids,
            device=device,
            chunk_size=args.score_chunk_size,
        )

        random_rxn = rng.choice(rxn_pool_ids)
        _, random_ph_norm, random_temp_norm = predict_pair(
            model=model,
            enzyme_vec=enzyme_vec,
            rxn_vec=rxn_vec_by_id[random_rxn],
            device=device,
        )

        pos_rxn = rng.choice(valid_pos)
        _, pos_ph_norm, pos_temp_norm = predict_pair(
            model=model,
            enzyme_vec=enzyme_vec,
            rxn_vec=rxn_vec_by_id[pos_rxn],
            device=device,
        )

        rows.append(
            {
                "seq_id": sample.seq_id,
                "true_ph": sample.true_ph,
                "true_temp": sample.true_temp,
                "top1_rxn_id": top1_rxn,
                "top1_pred_ph": top1_ph_norm * ph_std + ph_mean,
                "top1_pred_temp": top1_temp_norm * temp_std + temp_mean,
                "random_rxn_id": random_rxn,
                "random_pred_ph": random_ph_norm * ph_std + ph_mean,
                "random_pred_temp": random_temp_norm * temp_std + temp_mean,
                "pos_rxn_id": pos_rxn,
                "pos_pred_ph": pos_ph_norm * ph_std + ph_mean,
                "pos_pred_temp": pos_temp_norm * temp_std + temp_mean,
            }
        )

    if not rows:
        raise RuntimeError("No valid enzyme samples were evaluated.")

    result_df = pd.DataFrame(rows)

    def strategy_metrics(prefix: str):
        ph_mask = result_df["true_ph"].notna()
        temp_mask = result_df["true_temp"].notna()

        ph_mae, ph_rmse = calc_metrics(
            result_df.loc[ph_mask, f"{prefix}_pred_ph"].tolist(),
            result_df.loc[ph_mask, "true_ph"].tolist(),
        )
        temp_mae, temp_rmse = calc_metrics(
            result_df.loc[temp_mask, f"{prefix}_pred_temp"].tolist(),
            result_df.loc[temp_mask, "true_temp"].tolist(),
        )

        return {
            "ph_count": int(ph_mask.sum()),
            "ph_mae": ph_mae,
            "ph_rmse": ph_rmse,
            "temp_count": int(temp_mask.sum()),
            "temp_mae": temp_mae,
            "temp_rmse": temp_rmse,
        }

    top1_metrics = strategy_metrics("top1")
    random_metrics = strategy_metrics("random")
    pos_metrics = strategy_metrics("pos")

    print("\n=== Pairing Strategy Metrics (enzyme test split) ===")
    print(f"Total evaluated enzymes: {len(result_df)}")
    print(f"Skipped (missing enzyme embedding): {skipped_no_enzyme}")
    print(f"Skipped (no positive reaction embedding): {skipped_no_pos}")

    def print_strategy(name: str, metrics: Dict[str, float]):
        print(f"\n[{name}]")
        print(
            f"pH  -> n={metrics['ph_count']}, "
            f"MAE={metrics['ph_mae']:.4f}, RMSE={metrics['ph_rmse']:.4f}"
        )
        print(
            f"Temp-> n={metrics['temp_count']}, "
            f"MAE={metrics['temp_mae']:.4f}, RMSE={metrics['temp_rmse']:.4f}"
        )

    print_strategy("Top1 reaction", top1_metrics)
    print_strategy("Random reaction", random_metrics)
    print_strategy("Positive reaction", pos_metrics)

    if args.save_csv:
        out_dir = os.path.dirname(args.save_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-enzyme predictions: {args.save_csv}")


if __name__ == "__main__":
    main()
