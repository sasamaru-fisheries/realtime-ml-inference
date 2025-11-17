"""Titanicデータでランダムフォレストを訓練し、ONNX形式でエクスポートするスクリプト。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import yaml

DEFAULT_NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
DEFAULT_CATEGORICAL_COLUMNS = ["Sex", "Embarked"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic RandomForest and export ONNX.")
    parser.add_argument(
        "--csv",
        default="data/Titanic-Dataset.csv",
        help="学習データのCSVパス (default: data/Titanic-Dataset.csv)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="前処理・列定義を含むYAMLスキーマのパス（未指定時はTitanicデフォルト列を使用）",
    )
    parser.add_argument(
        "--onnx",
        default="models/titanic_random_forest.onnx",
        help="出力するONNXファイルパス",
    )
    parser.add_argument(
        "--pmml",
        default="models/titanic_random_forest.pmml",
        help="出力するPMMLファイルパス",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=200,
        help="RandomForestClassifierのn_estimators",
    )
    return parser.parse_args()


def load_schema(schema_path: str | None):
    if schema_path is None:
        return {
            "target": "Survived",
            "numeric": [{"name": c, "impute": "median"} for c in DEFAULT_NUMERIC_COLUMNS],
            "categorical": [{"name": c, "impute": "most_frequent", "encode": "onehot"} for c in DEFAULT_CATEGORICAL_COLUMNS],
        }
    return yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))


def load_and_preprocess(csv_path: str, schema: dict) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = pd.read_csv(csv_path)
    target = schema["target"]
    numeric_cols = [c["name"] for c in schema.get("numeric", [])]
    categorical_cols = [c["name"] for c in schema.get("categorical", [])]
    feature_cols = numeric_cols + categorical_cols
    df = df[[target, *feature_cols]].copy()

    # 数値列は欠損を中央値で埋める
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    # カテゴリ列は最頻値で埋める（全欠損なら空文字）
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        if df[col].isnull().all():
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    X = df[feature_cols]
    y = df[target].astype(np.int64).to_numpy()
    return X, y, numeric_cols, categorical_cols


def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, numeric_cols: list[str], categorical_cols: list[str]) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_pipeline = "passthrough"
    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=-1
            )),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    return clf


def export_onnx(model: RandomForestClassifier, numeric_cols: list[str], categorical_cols: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = []
    for col in numeric_cols:
        initial_type.append((col, FloatTensorType([None, 1])))
    for col in categorical_cols:
        initial_type.append((col, StringTensorType([None, 1])))
    options = {"zipmap": False}  # 出力をTensor形式にする
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
    output_path.write_bytes(onnx_model.SerializeToString())
    print(f"Saved ONNX model to {output_path}")


def export_pmml(model: Pipeline, feature_cols: list[str], output_path: Path, target: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pmml_pipeline = PMMLPipeline(model.steps)
    sklearn2pmml(pmml_pipeline, str(output_path), with_repr=True)
    print(f"Saved PMML model to {output_path}")


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    X, y, numeric_cols, categorical_cols = load_and_preprocess(args.csv, schema)
    model = train_model(X, y, args.estimators, numeric_cols, categorical_cols)
    feature_cols = numeric_cols + categorical_cols
    export_onnx(model, numeric_cols, categorical_cols, Path(args.onnx))
    export_pmml(model, feature_cols, Path(args.pmml), schema["target"])
    print("Feature order for inference:")
    for idx, name in enumerate(feature_cols):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
