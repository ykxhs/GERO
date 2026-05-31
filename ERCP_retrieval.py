import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from src.model import ERCP_FixRETR

class UniqueEnzymeDataset(Dataset):
    def __init__(self, unique_ids, gvp_dir, esm_data):
        self.ids = unique_ids
        self.gvp_dir = gvp_dir
        self.esm_data = esm_data
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        seq_id = str(self.ids[idx])
        gvp_path = os.path.join(self.gvp_dir, f"{seq_id}.pt")
        try:
            gvp_graph = torch.load(gvp_path, weights_only=False)
        except: return None
        esm_vec = self.esm_data.get(seq_id, torch.zeros(1280))
        return {'gvp_graph': gvp_graph, 'esm_emb': esm_vec, 'id': seq_id}

class UniqueRxnDataset(Dataset):
    def __init__(self, unique_ids, rxn_data, emb_dim):
        self.ids = unique_ids
        self.rxn_data = rxn_data
        self.emb_dim = emb_dim
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        rid = str(self.ids[idx])
        emb = self.rxn_data.get(rid, torch.zeros(self.emb_dim))
        return {'rxn_emb': emb, 'id': rid}

def enzyme_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    gvp_batch = Batch.from_data_list([b['gvp_graph'] for b in batch])
    esm_embs = torch.stack([b['esm_emb'] for b in batch])
    ids = [b['id'] for b in batch]
    return gvp_batch, esm_embs, ids

def rxn_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    rxn_embs = torch.stack([b['rxn_emb'] for b in batch])
    ids = [b['id'] for b in batch]
    return rxn_embs, ids

def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--data_path", type=str, default="/data2/caiyueyi/dataset/")
    parser.add_argument("--data_csv", type=str, default="/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv")
    parser.add_argument("--gvp_dir", type=str, default="/data2/caiyueyi/dataset/pocket_dataset/processed_tensors")
    parser.add_argument("--esm_emb", type=str, default="/data2/caiyueyi/dataset/esm_sequence_embeddings.pt")
    parser.add_argument("--rxn_emb", type=str, default="/data2/caiyueyi/dataset/rxn_unimol_embeddings.pt")
    parser.add_argument("--rxn_model", type=str, default="unimol", choices=["ChemBERTa", "rxnfp", "unimol"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model_ercp.pth")
    parser.add_argument("--output_dir", type=str, default="./results/score_matrix")

    # 模型超参
    parser.add_argument("--esm_dim", type=int, default=1280)
    parser.add_argument("--rxn_dim", type=int, default=768)
    parser.add_argument("--fusion_dim", type=int, default=256)

    # 运行参数
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--matrix_batch_size", type=int, default=128)
    parser.add_argument("--split_type", type=str, default="enzyme", choices=["enzyme", "reaction"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据元信息
    print(f"Loading CSV: {args.data_csv}")
    df = pd.read_csv(args.data_csv)

    unique_seq_ids = df['seq_id'].astype(str).unique().tolist()
    unique_rxn_ids = df['rxn_id'].astype(str).unique().tolist()
    print(f"Target: {len(unique_seq_ids)} Enzymes x {len(unique_rxn_ids)} Reactions")

    # 2. 加载 Embedding (静态文件)
    print("Loading Embeddings...")
    if args.rxn_model == "ChemBERTa":
        args.rxn_emb = os.path.join(args.data_path, "rxn_ChemBERTa_embeddings.pt")
        args.rxn_dim = 768
    if args.rxn_model == "rxnfp":  # rxnfp
        args.rxn_emb = os.path.join(args.data_path, "rxn_rxnfp_embeddings.pt")
        args.rxn_dim = 256
    if args.rxn_model == "unimol":  # unimol
        args.rxn_emb = os.path.join(args.data_path, "rxn_unimol_embeddings.pt")
        args.rxn_dim = 512

    esm_data = torch.load(args.esm_emb, map_location="cpu", weights_only=False)
    rxn_data = torch.load(args.rxn_emb, map_location="cpu", weights_only=False)
    esm_data = {str(k): v for k, v in esm_data.items()}
    rxn_data = {str(k): v for k, v in rxn_data.items()}

    # 3. 初始化模型
    print("Initializing Model...")
    model = ERCP_FixRETR(
        esm_dim=args.esm_dim,
        rxn_dim=args.rxn_dim,
        fusion_dim=args.fusion_dim
    )

    # 4. 加载权重
    print(f"Loading Weights: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # 5. 准备 DataLoaders
    enz_ds = UniqueEnzymeDataset(unique_seq_ids, args.gvp_dir, esm_data)
    rxn_ds = UniqueRxnDataset(unique_rxn_ids, rxn_data, args.rxn_dim)

    enz_loader = DataLoader(enz_ds, batch_size=args.batch_size, collate_fn=enzyme_collate, num_workers=4)
    rxn_loader = DataLoader(rxn_ds, batch_size=args.batch_size, collate_fn=rxn_collate, num_workers=4)

    # 6. Step 1: 预编码所有酶
    print("Encoding Enzymes...")
    all_enz_vecs = []
    final_enz_ids = []

    with torch.no_grad():
        for b in tqdm(enz_loader, desc="Enz Encode"):
            if b is None: continue
            gvp, esm, ids = b
            vec = model.encode_enzyme(gvp.to(device), esm.to(device))
            all_enz_vecs.append(vec.cpu())
            final_enz_ids.extend(ids)
    enz_tensor = torch.cat(all_enz_vecs, dim=0)

    # 7. Step 2: 预编码所有反应
    print("Encoding Reactions...")
    all_rxn_vecs = []
    final_rxn_ids = []

    with torch.no_grad():
        for b in tqdm(rxn_loader, desc="Rxn Encode"):
            if b is None: continue
            rxn_emb, ids = b
            vec = model.encode_reaction(rxn_emb.to(device))
            all_rxn_vecs.append(vec.cpu())
            final_rxn_ids.extend(ids)
    rxn_tensor = torch.cat(all_rxn_vecs, dim=0)

    # 8. Step 3: 矩阵计算 (Induced Fit 交互)
    print("Computing Interaction Matrix...")
    num_enz = enz_tensor.size(0)
    num_rxn = rxn_tensor.size(0)
    score_matrix = np.zeros((num_enz, num_rxn), dtype=np.float32)
    pred_pHopt_matrix = np.zeros((num_enz, num_rxn), dtype=np.float32)
    pred_topt_matrix = np.zeros((num_enz, num_rxn), dtype=np.float32)
    rxn_tensor = rxn_tensor.to(device) # [M, Dim]

    for i in tqdm(range(0, num_enz, args.matrix_batch_size), desc="Scoring"):
        # 取出一个 Batch 的酶
        e_batch = enz_tensor[i : i + args.matrix_batch_size].to(device) # [B, Dim]
        curr_bs = e_batch.size(0)
        e_exp = e_batch.unsqueeze(1).expand(-1, num_rxn, -1).reshape(-1, args.fusion_dim)
        r_exp = rxn_tensor.unsqueeze(0).expand(curr_bs, -1, -1).reshape(-1, args.fusion_dim)

        with torch.no_grad():
            fused = model.induced_fit(e_exp, r_exp)
            scores = model.predict_head(fused).squeeze(-1)
            pred_pHopt = model.predict_ph_head(fused).squeeze(-1)
            pred_topt = model.predict_temp_head(fused).squeeze(-1)

        score_matrix[i:i+curr_bs] = scores.view(curr_bs, num_rxn).cpu().numpy()
        pred_pHopt_matrix[i:i+curr_bs] = pred_pHopt.view(curr_bs, num_rxn).cpu().numpy()
        pred_topt_matrix[i:i+curr_bs] = pred_topt.view(curr_bs, num_rxn).cpu().numpy()

    # 9. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"ERCP_{args.split_type}_split_score_matrix.npy")
    np.save(save_path, {
        'score_matrix': score_matrix,
        'pred_pHopt_matrix': pred_pHopt_matrix,
        'pred_topt_matrix': pred_topt_matrix,
        'row_ids': final_enz_ids,
        'col_ids': final_rxn_ids
    })
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
