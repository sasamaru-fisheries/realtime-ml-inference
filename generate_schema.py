"""CSVから簡易スキーマ(YAML)を推定して出力するスクリプト。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer schema (numeric/categorical) from CSV and dump YAML.")
    p.add_argument("--csv", required=True, help="入力CSVパス")
    p.add_argument("--target", required=True, help="目的変数の列名")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="除外したい列（ID列など）スペース区切りで指定",
    )
    p.add_argument(
        "--output",
        default="schema.yaml",
        help="出力するYAMLパス (default: schema.yaml)",
    )
    return p.parse_args()


def infer_schema(csv_path: str, target: str, exclude: list[str]) -> dict:
    df = pd.read_csv(csv_path, nrows=5000)  # サンプリングして型推定
    cols = [c for c in df.columns if c not in exclude and c != target]
    numeric = []
    categorical = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {
        "target": target,
        "numeric": [{"name": c, "impute": "median"} for c in numeric],
        "categorical": [{"name": c, "impute": "most_frequent", "encode": "onehot"} for c in categorical],
    }


def main() -> None:
    args = parse_args()
    schema = infer_schema(args.csv, args.target, args.exclude)
    out_path = Path(args.output)
    out_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    print(f"Wrote schema to {out_path}")
    print("Edit this file if必要に応じて個別列の設定を上書きしてください。")


if __name__ == "__main__":
    main()
