import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm


if __package__ is None or __package__ == "":
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

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



def check_prop_valid(row) -> bool:
	if pd.isna(row["ph_opt"]) or pd.isna(row["temp_opt"]):
		return False
	return float(row["ph_opt"]) > 0 and float(row["temp_opt"]) > 0



def collect_truth_info(raw_df: pd.DataFrame, query_col: str, candidate_col: str, query_id: str):
	subset = raw_df[raw_df[query_col].astype(str) == str(query_id)].copy()
	if subset.empty:
		return {}

	subset[candidate_col] = subset[candidate_col].astype(str)
	truth_info = {}
	for candidate_id, group in subset.groupby(candidate_col):
		valid_prop_rows = [
			{"ph": float(row["ph_opt"]), "temp": float(row["temp_opt"])}
			for _, row in group.iterrows()
			if check_prop_valid(row)
		]
		truth_info[str(candidate_id)] = {
			"is_true": True,
			"pair_count": int(len(group)),
			"true_ph": valid_prop_rows[0]["ph"] if valid_prop_rows else None,
			"true_temp": valid_prop_rows[0]["temp"] if valid_prop_rows else None,
		}
	return truth_info



def infer_split_from_checkpoint(checkpoint_path: str) -> str:
	checkpoint_name = os.path.basename(checkpoint_path).lower()
	if "reaction" in checkpoint_name:
		return "reaction"
	return "enzyme"



def load_single_enzyme(seq_id: str, gvp_dir: str, esm_data: Dict[str, torch.Tensor]):
	gvp_path = os.path.join(gvp_dir, f"{seq_id}.pt")
	if not os.path.exists(gvp_path):
		raise FileNotFoundError(f"GVP graph not found for seq_id={seq_id}: {gvp_path}")

	gvp_graph = torch.load(gvp_path, map_location="cpu", weights_only=False)
	esm_vec = esm_data.get(seq_id)
	if esm_vec is None:
		raise KeyError(f"ESM embedding not found for seq_id={seq_id}")

	return gvp_graph, torch.as_tensor(esm_vec, dtype=torch.float32)



def load_single_reaction(rxn_id: str, rxn_data: Dict[str, torch.Tensor]) -> torch.Tensor:
	rxn_vec = rxn_data.get(rxn_id)
	if rxn_vec is None:
		raise KeyError(f"Reaction embedding not found for rxn_id={rxn_id}")
	return torch.as_tensor(rxn_vec, dtype=torch.float32)


@torch.no_grad()
def encode_enzyme_candidates(
	model,
	candidate_ids: List[str],
	gvp_dir: str,
	esm_data: Dict[str, torch.Tensor],
	device: torch.device,
	batch_size: int,
	num_workers: int,
):
	dataset = UniqueEnzymeDataset(candidate_ids, gvp_dir, esm_data)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=enzyme_collate,
	)

	encoded_parts = []
	valid_ids = []
	for batch in tqdm(loader, desc="Encode candidate enzymes"):
		if batch is None:
			continue
		gvp_batch, esm_batch, ids = batch
		vecs = model.encode_enzyme(gvp_batch.to(device), esm_batch.to(device))
		encoded_parts.append(vecs.cpu())
		valid_ids.extend(ids)

	if not encoded_parts:
		raise RuntimeError("No valid enzyme candidates were encoded.")

	return torch.cat(encoded_parts, dim=0), valid_ids


@torch.no_grad()
def encode_reaction_candidates(
	model,
	candidate_ids: List[str],
	rxn_data: Dict[str, torch.Tensor],
	device: torch.device,
	batch_size: int,
	num_workers: int,
):
	dataset = UniqueRxnDataset(candidate_ids, rxn_data)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=rxn_collate,
	)

	encoded_parts = []
	valid_ids = []
	for batch in tqdm(loader, desc="Encode candidate reactions"):
		if batch is None:
			continue
		rxn_batch, ids = batch
		vecs = model.encode_reaction(rxn_batch.to(device))
		encoded_parts.append(vecs.cpu())
		valid_ids.extend(ids)

	if not encoded_parts:
		raise RuntimeError("No valid reaction candidates were encoded.")

	return torch.cat(encoded_parts, dim=0), valid_ids


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
	for index in range(0, candidate_vecs.size(0), chunk_size):
		candidates = candidate_vecs[index : index + chunk_size].to(device)
		current_size = candidates.size(0)

		if query_is_enzyme:
			e_flat = q.expand(current_size, -1)
			r_flat = candidates
		else:
			e_flat = candidates
			r_flat = q.expand(current_size, -1)

		fused = model.induced_fit(e_flat, r_flat)
		score_parts.append(model.predict_head(fused).squeeze(-1).cpu())
		ph_parts.append(model.predict_ph_head(fused).squeeze(-1).cpu())
		temp_parts.append(model.predict_temp_head(fused).squeeze(-1).cpu())

	scores = torch.cat(score_parts, dim=0).numpy()
	ph_norm = torch.cat(ph_parts, dim=0).numpy()
	temp_norm = torch.cat(temp_parts, dim=0).numpy()
	return scores, ph_norm, temp_norm



def build_result_dataframe(
	candidate_ids: List[str],
	scores: np.ndarray,
	ph_norm: np.ndarray,
	temp_norm: np.ndarray,
	ph_mean: float,
	ph_std: float,
	temp_mean: float,
	temp_std: float,
	truth_info: Dict[str, Dict[str, Optional[float]]],
):
	sorted_indices = np.argsort(-scores)
	rows = []
	for rank, candidate_index in enumerate(sorted_indices, start=1):
		candidate_id = candidate_ids[int(candidate_index)]
		true_item = truth_info.get(candidate_id, {})
		rows.append(
			{
				"rank": rank,
				"candidate_id": candidate_id,
				"score": float(scores[candidate_index]),
				"pred_ph": float(ph_norm[candidate_index]) * ph_std + ph_mean,
				"pred_temp": float(temp_norm[candidate_index]) * temp_std + temp_mean,
				"is_true_sample": bool(true_item.get("is_true", False)),
				"true_ph": true_item.get("true_ph"),
				"true_temp": true_item.get("true_temp"),
			}
		)
	return pd.DataFrame(rows)



def print_top_results(results_df: pd.DataFrame, top_k: int, query_label: str, candidate_label: str):
	top_df = results_df.head(top_k)
	print(f"\nQuery: {query_label}")
	print(f"Top-{min(top_k, len(results_df))} ranked {candidate_label} candidates:")
	if top_df.empty:
		print("No results.")
		return

	print(
		top_df.to_string(
			index=False,
			columns=["rank", "candidate_id", "score", "pred_ph", "pred_temp", "is_true_sample"],
		)
	)



def print_truth_samples(truth_info: Dict[str, Dict[str, Optional[float]]], candidate_label: str):
	true_ids = list(truth_info.keys())
	print(f"\nTrue {candidate_label} samples for this query: {len(true_ids)}")
	if not true_ids:
		print("None")
		return
	print(", ".join(true_ids))



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



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--split", type=str, default=None, choices=["enzyme", "reaction"])

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
	parser.add_argument("--top_k", type=int, default=10)

	query_group = parser.add_mutually_exclusive_group(required=True)
	query_group.add_argument("--seq_id", type=str, default=None)
	query_group.add_argument("--rxn_id", type=str, default=None)
	return parser.parse_args()



def main():
	args = parse_args()
	if args.split is None:
		args.split = infer_split_from_checkpoint(args.checkpoint)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	raw_df = pd.read_csv(args.data_csv)
	raw_df["seq_id"] = raw_df["seq_id"].astype(str)
	raw_df["rxn_id"] = raw_df["rxn_id"].astype(str)

	train_df = raw_df[raw_df[f"{args.split}_set"] == "train"].copy()
	if train_df.empty:
		raise RuntimeError(f"No train rows found for split={args.split}")

	ph_mean = float(train_df["ph_opt"].mean())
	ph_std = float(train_df["ph_opt"].std())
	temp_mean = float(train_df["temp_opt"].mean())
	temp_std = float(train_df["temp_opt"].std())

	rxn_emb_path, rxn_dim = get_rxn_emb_path(args.data_path, args.rxn_model, args.rxn_emb)
	print(f"Loading ESM embeddings: {args.esm_emb}")
	print(f"Loading reaction embeddings: {rxn_emb_path}")

	esm_data = torch.load(args.esm_emb, map_location="cpu", weights_only=False)
	rxn_data = torch.load(rxn_emb_path, map_location="cpu", weights_only=False)
	esm_data = {str(key): value for key, value in esm_data.items()}
	rxn_data = {str(key): value for key, value in rxn_data.items()}

	state_dict = load_checkpoint_state_dict(args.checkpoint)
	model, inferred = build_model(args, state_dict, rxn_dim, device)
	print(
		f"Loaded checkpoint: {args.checkpoint} "
		f"(esm_dim={inferred['esm_dim']}, rxn_dim={inferred['rxn_dim']}, "
		f"fusion_dim={inferred['fusion_dim']}, gvp_layers={inferred['gvp_layers']}, "
		f"rxn_blocks={inferred['rxn_blocks']})"
	)

	if args.seq_id:
		query_id = str(args.seq_id)
		query_type = "enzyme"
		candidate_label = "reaction"
		truth_info = collect_truth_info(raw_df, "seq_id", "rxn_id", query_id)
		gvp_graph, esm_vec = load_single_enzyme(query_id, args.gvp_dir, esm_data)
		query_batch = Batch.from_data_list([gvp_graph])
		query_vec = model.encode_enzyme(query_batch.to(device), esm_vec.unsqueeze(0).to(device)).cpu()

		candidate_ids = raw_df["rxn_id"].astype(str).unique().tolist()
		candidate_vecs, valid_candidate_ids = encode_reaction_candidates(
			model=model,
			candidate_ids=candidate_ids,
			rxn_data=rxn_data,
			device=device,
			batch_size=args.encode_batch_size,
			num_workers=args.num_workers,
		)
		scores, ph_norm, temp_norm = score_query_to_candidates(
			model=model,
			query_vec=query_vec,
			candidate_vecs=candidate_vecs,
			query_is_enzyme=True,
			device=device,
			chunk_size=args.score_chunk_size,
		)
	else:
		query_id = str(args.rxn_id)
		query_type = "reaction"
		candidate_label = "enzyme"
		truth_info = collect_truth_info(raw_df, "rxn_id", "seq_id", query_id)
		rxn_vec = load_single_reaction(query_id, rxn_data)
		query_vec = model.encode_reaction(rxn_vec.unsqueeze(0).to(device)).cpu()

		candidate_ids = raw_df["seq_id"].astype(str).unique().tolist()
		candidate_vecs, valid_candidate_ids = encode_enzyme_candidates(
			model=model,
			candidate_ids=candidate_ids,
			gvp_dir=args.gvp_dir,
			esm_data=esm_data,
			device=device,
			batch_size=args.encode_batch_size,
			num_workers=args.num_workers,
		)
		scores, ph_norm, temp_norm = score_query_to_candidates(
			model=model,
			query_vec=query_vec,
			candidate_vecs=candidate_vecs,
			query_is_enzyme=False,
			device=device,
			chunk_size=args.score_chunk_size,
		)

	results_df = build_result_dataframe(
		candidate_ids=valid_candidate_ids,
		scores=scores,
		ph_norm=ph_norm,
		temp_norm=temp_norm,
		ph_mean=ph_mean,
		ph_std=ph_std,
		temp_mean=temp_mean,
		temp_std=temp_std,
		truth_info=truth_info,
	)
	print(f"Candidate pool size: {len(results_df)}")
	print_top_results(results_df, args.top_k, f"{query_type}_id={query_id}", candidate_label)
	print_truth_samples(truth_info, candidate_label)


if __name__ == "__main__":
	main()
