#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${DEMO_DIR}/.." && pwd)"
DATA_DIR="${DEMO_DIR}/demo_data"
OUT_DIR="${DEMO_DIR}/demo_outputs"

python "${DEMO_DIR}/make_demo_data.py" \
  --out_dir "${DATA_DIR}" \
  --seed 13 \
  --esm_dim 64 \
  --rxn_model rxnfp

python "${ROOT_DIR}/train_fix_rxn.py" \
  --data_path "${DATA_DIR}" \
  --gvp_dir "${DATA_DIR}/pocket_dataset/processed_tensors" \
  --esm_emb "${DATA_DIR}/esm_sequence_embeddings.pt" \
  --rxn_model rxnfp \
  --split_type enzyme \
  --model_name demo_toy \
  --esm_dim 64 \
  --fusion_dim 32 \
  --batch_size 2 \
  --epochs 1 \
  --warmup_epochs 1 \
  --eval_every 1 \
  --step3_start_epoch 999 \
  --save_dir "${OUT_DIR}/checkpoints" \
  --log_dir "${OUT_DIR}/logs"

python "${ROOT_DIR}/evaluate_retrieval_fix.py" \
  --checkpoint "${OUT_DIR}/checkpoints/latest_model_demo_toy.pth" \
  --split enzyme \
  --task all \
  --data_path "${DATA_DIR}" \
  --gvp_dir "${DATA_DIR}/pocket_dataset/processed_tensors" \
  --esm_emb "${DATA_DIR}/esm_sequence_embeddings.pt" \
  --rxn_model rxnfp \
  --encode_batch_size 2 \
  --score_chunk_size 32 \
  --matrix_batch_size 4 \
  --save_json "${OUT_DIR}/metrics.json" \
  --save_matrix ""

echo "Demo finished. Outputs in ${OUT_DIR}"