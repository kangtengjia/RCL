#!/usr/bin/env python3
"""Export RCL-SGRAF RoMa scores in the canonical HOAR coarse format."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--data-root", required=True)
parser.add_argument("--query-jsonl", required=True)
parser.add_argument("--output-jsonl", required=True)
parser.add_argument("--metadata-json", required=True)
parser.add_argument("--top-k", type=int, default=50)
args, rest = parser.parse_known_args()
subprocess.run([sys.executable, str(ROOT / "tools/export_comparison_hoar_coarse.py"), "--method", "RCL_SGRAF", "--checkpoint", args.checkpoint, "--dataset", args.dataset, "--data-root", args.data_root, "--query-jsonl", args.query_jsonl, "--output-jsonl", args.output_jsonl, "--metadata-json", args.metadata_json, "--top-k", str(args.top_k), *rest], check=True)
