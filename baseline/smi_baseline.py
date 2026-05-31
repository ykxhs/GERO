import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data2/caiyueyi/dataset")
    parser.add_argument("--esm_path", type=str, default="/data2/caiyueyi/dataset/esm_sequence_embeddings.pt")
    parser.add_argument("--rxn_path", type=str, default=None)
    parser.add_argument("--rxn_model", type=str, default="unimol", choices=["ChemBERTa", "rxnfp", "unimol"])
    parser.add_argument("--raw_csv", type=str, default="/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv")
    parser.add_argument("--output_dir", type=str, default="/data2/caiyueyi/RepCode/R_results/score_matrix")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(">>> Loading embeddings...")
    if args.rxn_path is None:
        if args.rxn_model == "ChemBERTa":
            args.rxn_path = os.path.join(args.data_path, "rxn_ChemBERTa_embeddings.pt")
        elif args.rxn_model == "rxnfp":
            args.rxn_path = os.path.join(args.data_path, "rxn_rxnfp_embeddings.pt")
        else:
            args.rxn_path = os.path.join(args.data_path, "rxn_unimol_embeddings.pt")

    esm_data = torch.load(args.esm_path, map_location="cpu")
    rxn_data = torch.load(args.rxn_path, map_location="cpu")
    print(f"    Loaded {len(esm_data)} enzyme embeddings and {len(rxn_data)} reaction embeddings.")

    df = pd.read_csv(args.raw_csv)
    unique_seq_ids = df["seq_id"].unique().astype(str).tolist()
    unique_rxn_ids = df["rxn_id"].unique().astype(str).tolist()

    print(">>> Stacking features...")
    valid_seq_ids = [i for i in unique_seq_ids if i in esm_data]
    valid_rxn_ids = [i for i in unique_rxn_ids if i in rxn_data]

    E_raw = np.stack([esm_data[i].numpy() for i in valid_seq_ids])
    R_raw = np.stack([rxn_data[i].numpy() for i in valid_rxn_ids])

    print(f"    Enzymes: {E_raw.shape}, Reactions: {R_raw.shape}")

    print(">>> Running PCA to reduce enzymes to 768 dim...")
    pca = PCA(n_components=768)
    E_proj = pca.fit_transform(E_raw)
    R_proj = R_raw

    print(">>> Normalizing features (cosine similarity pre-step)...")
    E_proj = normalize(E_proj, axis=1, norm="l2")
    R_proj = normalize(R_proj, axis=1, norm="l2")

    print(">>> Computing baseline score matrix (dot product)...")
    score_matrix = np.matmul(E_proj, R_proj.T)

    save_dict = {
        "score_matrix": score_matrix,
        "row_ids": valid_seq_ids,
        "col_ids": valid_rxn_ids,
    }

    output_filename = "baselinesmi_enzyme_split_score_matrix.npy"
    output_file = os.path.join(args.output_dir, output_filename)
    np.save(output_file, save_dict)

    print(f"Baseline matrix saved to {output_file}")
    print(f"Shape: {score_matrix.shape}")
    print("Note: expected performance is weak due to missing alignment.")


if __name__ == "__main__":
    main()
