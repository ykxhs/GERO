Mini Demo
=========

This demo generates a tiny synthetic dataset and runs a 1-epoch training + evaluation.
It is only a pipeline smoke test; numbers are not meaningful.

Requirements
------------
- Python dependencies from requirements.txt
- torch-geometric + torch-cluster installed for your PyTorch/CUDA

Run
---

From anywhere:

```bash
bash /data2/caiyueyi/ercp/.gero_upload_GuTCL4/demo/run_demo.sh
```

Outputs
-------
- demo/demo_data/ contains synthetic CSV, embeddings, and pocket graphs
- demo/demo_outputs/ contains a checkpoint and metrics JSON
