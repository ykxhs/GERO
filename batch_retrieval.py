import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SPLITS = ["enzyme", "reaction"]
DEFAULT_METHODS = ["ChemBERTa", "rxnfp", "woGate"]


def method_to_rxn_model(method: str) -> str:
	if method == "ChemBERTa":
		return "ChemBERTa"
	if method == "rxnfp":
		return "rxnfp"
	return "unimol"


def build_command(
	python_exec: str,
	ercp_script: Path,
	checkpoint_path: Path,
	split: str,
	method: str,
	save_json_path: Path,
	score_chunk_size: int,
	matrix_batch_size: int,
):
	rxn_model = method_to_rxn_model(method)

	cmd = [
		python_exec,
		str(ercp_script),
		"--checkpoint",
		str(checkpoint_path),
		"--split",
		split,
		"--task",
		"all",
		"--rxn_model",
		rxn_model,
		"--save_json",
		str(save_json_path),
		"--score_chunk_size",
		str(score_chunk_size),
		"--matrix_batch_size",
		str(matrix_batch_size),
	]
	return cmd


def main():
	parser = argparse.ArgumentParser(
		description="Batch retrieval for ERCP across split/method combinations."
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=DEFAULT_SPLITS,
		choices=DEFAULT_SPLITS,
		help="Dataset split modes to evaluate.",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		default=DEFAULT_METHODS,
		choices=DEFAULT_METHODS,
		help="Ablation methods in checkpoint suffix.",
	)
	parser.add_argument(
		"--checkpoint_dir",
		type=str,
		default="/data2/caiyueyi/ercp/checkpoints/ablation_study",
		help="Directory containing best_model_ercp_{split}_smi_{method}.pth checkpoints.",
	)
	parser.add_argument(
		"--checkpoint_pattern",
		type=str,
		default="best_model_ercp_{split}_smi_{method}.pth",
		help="Checkpoint filename pattern.",
	)
	parser.add_argument(
		"--ercp_script",
		type=str,
		default="/data2/caiyueyi/ercp/ERCP_retrieval.py",
		help="Path to ERCP retrieval script.",
	)
	parser.add_argument(
		"--result_dir",
		type=str,
		default="/data2/caiyueyi/ercp/results/retrieval_batch",
		help="Directory to save json summaries.",
	)
	parser.add_argument(
		"--python",
		type=str,
		default=sys.executable,
		help="Python executable used to run ERCP_retrieval.py.",
	)
	parser.add_argument(
		"--score_chunk_size",
		type=int,
		default=2048,
		help="Forwarded to ERCP_retrieval.py --score_chunk_size.",
	)
	parser.add_argument(
		"--matrix_batch_size",
		type=int,
		default=256,
		help="Forwarded to ERCP_retrieval.py --matrix_batch_size.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Only print commands without executing.",
	)

	args = parser.parse_args()

	checkpoint_dir = Path(args.checkpoint_dir)
	ercp_script = Path(args.ercp_script)
	result_dir = Path(args.result_dir)

	if not ercp_script.exists():
		raise FileNotFoundError(f"ERCP script not found: {ercp_script}")

	result_dir.mkdir(parents=True, exist_ok=True)

	total = len(args.splits) * len(args.methods)
	done = 0
	failed = []

	print(f"Total tasks: {total}")
	print(f"Splits: {args.splits}")
	print(f"Methods: {args.methods}")

	for split in args.splits:
		for method in args.methods:
			ckpt_name = args.checkpoint_pattern.format(split=split, method=method)
			ckpt_path = checkpoint_dir / ckpt_name
			if not ckpt_path.exists():
				print(f"[Skip] checkpoint not found: {ckpt_path}")
				failed.append((split, method, "checkpoint_not_found"))
				continue

			save_json = result_dir / f"ERCP_{split}_{method}_retrieval.json"
			cmd = build_command(
				python_exec=args.python,
				ercp_script=ercp_script,
				checkpoint_path=ckpt_path,
				split=split,
				method=method,
				save_json_path=save_json,
				score_chunk_size=args.score_chunk_size,
				matrix_batch_size=args.matrix_batch_size,
			)

			done += 1
			print("\n" + "=" * 80)
			print(f"[{done}/{total}] split={split} method={method}")
			print("Command:", " ".join(cmd))

			if args.dry_run:
				continue

			proc = subprocess.run(cmd, check=False)
			if proc.returncode != 0:
				print(f"[Fail] split={split}, method={method}, code={proc.returncode}")
				failed.append((split, method, f"exit_code_{proc.returncode}"))
			else:
				print(f"[OK] result json: {save_json}")

	print("\n" + "=" * 80)
	print("Batch retrieval finished.")
	if failed:
		print(f"Failed/Skipped tasks: {len(failed)}")
		for split, method, reason in failed:
			print(f"- split={split}, method={method}, reason={reason}")
		raise SystemExit(1)

	print("All tasks completed successfully.")


if __name__ == "__main__":
	main()
