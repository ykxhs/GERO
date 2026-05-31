import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

# ================= Configuration =================
DATA_PATH = "/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv"
OUTPUT_PATH = "/data2/caiyueyi/RepCode/R_results/score_matrix/Baseline_reaction_split_score_matrix_task2.npy"


def standardize_molecule_smiles(mol_smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return mol_smiles
        rdmolops.SanitizeMol(mol, sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_CLEANUP)
        disconnector = rdMolStandardize.MetalDisconnector()
        mol = disconnector.Disconnect(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        rdmolops.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return mol_smiles


def create_transformation_FP(rxn: str, radius: int = 3):
    """Create reaction fingerprints by concatenating reactant and product FPs."""
    rfp = None
    pfp = None
    if ">>" not in rxn:
        return None

    reactants, products = rxn.split(">>")

    for react in filter(None, reactants.split(".")):
        mol = Chem.MolFromSmiles(react)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol=mol, radius=radius)
            rfp = fp if rfp is None else rfp + fp

    for product in filter(None, products.split(".")):
        mol = Chem.MolFromSmiles(product)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol=mol, radius=radius)
            pfp = fp if pfp is None else pfp + fp

    if pfp is not None and rfp is not None:
        return pfp + rfp
    if pfp is not None:
        return pfp
    if rfp is not None:
        return rfp
    return None


def compute_similarity_batch(test_fps, train_fps):
    """Compute Tanimoto similarity between test and train fingerprints."""
    results = []
    for i, q_fp in enumerate(tqdm(test_fps, desc="Calculating Similarity")):
        if q_fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
        for j, score in enumerate(sims):
            if score > 0:
                results.append((i, j, score))
    return results


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    df["rxn_id"] = df["rxn_id"].astype(str)
    df["seq_id"] = df["seq_id"].astype(str)

    train_df = df[df["reaction_set"] == "train"].copy()
    test_df = df[df["reaction_set"] == "test"].copy()

    all_test_rxn_ids = sorted(list(test_df["rxn_id"].unique()))
    all_seq_ids = sorted(list(df["seq_id"].unique()))

    print(f"Test reactions: {len(all_test_rxn_ids)}")
    print(f"Candidate enzymes: {len(all_seq_ids)}")
    print(f"Target matrix shape: ({len(all_test_rxn_ids)}, {len(all_seq_ids)})")

    train_rxn_unique = train_df[["rxn_id", "rxn_smiles"]].drop_duplicates(subset=["rxn_id"])
    train_rxn_ids_list = train_rxn_unique["rxn_id"].tolist()
    train_rxn_smiles_list = train_rxn_unique["rxn_smiles"].tolist()

    test_rxn_unique = test_df[["rxn_id", "rxn_smiles"]].drop_duplicates(subset=["rxn_id"])
    test_rxn_ids_list = test_rxn_unique["rxn_id"].tolist()
    test_rxn_smiles_list = test_rxn_unique["rxn_smiles"].tolist()

    print("Computing train reaction fingerprints...")
    train_fps = []
    valid_train_indices = []
    for idx, smi in enumerate(tqdm(train_rxn_smiles_list)):
        fp = create_transformation_FP(smi)
        if fp is not None:
            train_fps.append(fp)
            valid_train_indices.append(idx)

    print("Computing test reaction fingerprints...")
    test_fps = []
    for smi in tqdm(test_rxn_smiles_list):
        test_fps.append(create_transformation_FP(smi))

    sim_results = compute_similarity_batch(test_fps, train_fps)

    print("Processing similarity results...")

    train_fps_idx_to_rxnid = np.array([train_rxn_ids_list[i] for i in valid_train_indices])
    test_fps_idx_to_rxnid = np.array(test_rxn_ids_list)

    if not sim_results:
        print("Warning: no similarity found, matrix will be all zeros.")
        final_matrix = np.zeros((len(all_test_rxn_ids), len(all_seq_ids)), dtype=np.float32)
    else:
        test_indices = [x[0] for x in sim_results]
        train_indices = [x[1] for x in sim_results]
        scores = [x[2] for x in sim_results]

        sim_df = pd.DataFrame(
            {
                "test_rxn_id": test_fps_idx_to_rxnid[test_indices],
                "train_rxn_id": train_fps_idx_to_rxnid[train_indices],
                "similarity": scores,
            }
        )

        merged = pd.merge(
            sim_df,
            train_df[["rxn_id", "seq_id"]],
            left_on="train_rxn_id",
            right_on="rxn_id",
            how="inner",
        )

        print("Grouping scores...")
        score_df = merged.groupby(["test_rxn_id", "seq_id"])["similarity"].max().reset_index()
        score_df.rename(columns={"similarity": "score"}, inplace=True)

        print("Preparing matrix indices...")
        row_idx_map = {rid: i for i, rid in enumerate(all_test_rxn_ids)}
        col_idx_map = {sid: i for i, sid in enumerate(all_seq_ids)}

        final_matrix = np.zeros((len(all_test_rxn_ids), len(all_seq_ids)), dtype=np.float32)

        valid_mask = score_df["test_rxn_id"].isin(row_idx_map) & score_df["seq_id"].isin(col_idx_map)
        filtered_scores = score_df[valid_mask].copy()

        print("Mapping indices (vectorized)...")
        r_indices = filtered_scores["test_rxn_id"].map(row_idx_map).values
        c_indices = filtered_scores["seq_id"].map(col_idx_map).values
        score_values = filtered_scores["score"].values

        print("Filling final matrix (vectorized)...")
        final_matrix[r_indices, c_indices] = score_values

    save_dict = {
        "score_matrix": final_matrix.T,
        "row_ids": all_seq_ids,
        "col_ids": all_test_rxn_ids,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, save_dict)

    print(f"Matrix shape: {final_matrix.T.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
