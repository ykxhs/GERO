import pandas as pd
import numpy as np
import os
import subprocess
import shutil
from tqdm import tqdm

# ================= 						 										
DATA_PATH = "/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv"
WORK_DIR = "./mmseqs_baseline_work"
OUTPUT_PATH = "/data2/caiyueyi/RepCode/R_results/score_matrix/Baseline_enzyme_split_score_matrix_task1.npy"
# ================= 														

def create_fasta(df, output_path):
    unique_df = df[["seq_id", "seq"]].drop_duplicates(subset=["seq_id"])

    with open(output_path, "w") as f:
        for _, row in unique_df.iterrows():
            clean_seq = row["seq"].strip()
            f.write(f">{row['seq_id']}\n{clean_seq}\n")
    return unique_df["seq_id"].tolist()


def run_mmseqs(query_fasta, target_fasta, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    target_db = os.path.join(work_dir, "targetDB")
    query_db = os.path.join(work_dir, "queryDB")
    result_db = os.path.join(work_dir, "resultDB")
    result_tsv = os.path.join(work_dir, "result.tsv")
    tmp_dir = os.path.join(work_dir, "tmp")

    if not os.path.exists(target_db + ".dbtype"):
        subprocess.run(
            ["mmseqs", "createdb", target_fasta, target_db],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    import os
    import shutil
    import subprocess

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # ================= Configuration =================
    DATA_PATH = "/data2/caiyueyi/dataset/enzyme_reaction_data_splits.csv"
    WORK_DIR = "./mmseqs_baseline_work"
    OUTPUT_PATH = "/data2/caiyueyi/RepCode/R_results/score_matrix/Baseline_enzyme_split_score_matrix_task1.npy"


    def create_fasta(df, output_path):
        unique_df = df[["seq_id", "seq"]].drop_duplicates(subset=["seq_id"])
        with open(output_path, "w") as f:
            for _, row in unique_df.iterrows():
                clean_seq = row["seq"].strip()
                f.write(f">{row['seq_id']}\n{clean_seq}\n")
        return unique_df["seq_id"].tolist()


    def run_mmseqs(query_fasta, target_fasta, work_dir):
        os.makedirs(work_dir, exist_ok=True)

        target_db = os.path.join(work_dir, "targetDB")
        query_db = os.path.join(work_dir, "queryDB")
        result_db = os.path.join(work_dir, "resultDB")
        result_tsv = os.path.join(work_dir, "result.tsv")
        tmp_dir = os.path.join(work_dir, "tmp")

        if not os.path.exists(target_db + ".dbtype"):
            subprocess.run(
                ["mmseqs", "createdb", target_fasta, target_db],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        subprocess.run(
            ["mmseqs", "createdb", query_fasta, query_db],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "mmseqs",
                "search",
                query_db,
                target_db,
                result_db,
                tmp_dir,
                "-s",
                "7.5",
                "-c",
                "0.8",
                "--cov-mode",
                "1",
                "--max-seqs",
                "200",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "mmseqs",
                "convertalis",
                query_db,
                target_db,
                result_db,
                result_tsv,
                "--format-output",
                "query,target,fident",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        return result_tsv


    def main():
        if os.path.exists(WORK_DIR):
            shutil.rmtree(WORK_DIR)

        df = pd.read_csv(DATA_PATH)
        train_df = df[df["enzyme_set"] == "train"].copy()
        test_df = df[df["enzyme_set"] == "test"].copy()

        all_rxn_ids = sorted(list(df["rxn_id"].unique()))
        all_test_seq_ids = sorted(list(test_df["seq_id"].unique()))

        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        print(f"Matrix shape: ({len(all_test_seq_ids)}, {len(all_rxn_ids)})")

        os.makedirs(WORK_DIR, exist_ok=True)
        create_fasta(train_df, os.path.join(WORK_DIR, "train.fasta"))
        create_fasta(test_df, os.path.join(WORK_DIR, "test.fasta"))

        result_tsv = run_mmseqs(
            os.path.join(WORK_DIR, "test.fasta"),
            os.path.join(WORK_DIR, "train.fasta"),
            WORK_DIR,
        )

        try:
            sim_df = pd.read_csv(result_tsv, sep="\t", names=["test_seq_id", "train_seq_id", "similarity"])
        except pd.errors.EmptyDataError:
            print("Warning: no homologous match found, matrix will be all zeros.")
            sim_df = pd.DataFrame(columns=["test_seq_id", "train_seq_id", "similarity"])

        sim_df["test_seq_id"] = sim_df["test_seq_id"].astype(str)
        sim_df["train_seq_id"] = sim_df["train_seq_id"].astype(str)
        train_df["seq_id"] = train_df["seq_id"].astype(str)

        merged = pd.merge(
            sim_df,
            train_df[["seq_id", "rxn_id"]],
            left_on="train_seq_id",
            right_on="seq_id",
            how="inner",
        )

        score_df = merged.groupby(["test_seq_id", "rxn_id"])["similarity"].max().reset_index()
        score_df.rename(columns={"similarity": "score"}, inplace=True)

        row_idx_map = {sid: i for i, sid in enumerate(all_test_seq_ids)}
        col_idx_map = {rid: i for i, rid in enumerate(all_rxn_ids)}

        final_matrix = np.zeros((len(all_test_seq_ids), len(all_rxn_ids)), dtype=np.float32)
        valid_mask = score_df["rxn_id"].isin(col_idx_map) & score_df["test_seq_id"].isin(row_idx_map)
        filtered_scores = score_df[valid_mask]

        for _, row in tqdm(filtered_scores.iterrows(), total=len(filtered_scores), desc="Filling Matrix"):
            r_idx = row_idx_map[row["test_seq_id"]]
            c_idx = col_idx_map[row["rxn_id"]]
            final_matrix[r_idx, c_idx] = row["score"]

        save_dict = {
            "score_matrix": final_matrix,
            "row_ids": all_test_seq_ids,
            "col_ids": all_rxn_ids,
        }

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        np.save(OUTPUT_PATH, save_dict)

        print(f"Matrix shape: {final_matrix.shape}")
        if os.path.exists(WORK_DIR):
            shutil.rmtree(WORK_DIR)


    if __name__ == "__main__":
        main()
