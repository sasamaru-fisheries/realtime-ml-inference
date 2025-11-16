"""Titanicデータでランダムフォレストを訓練し、ONNX形式でエクスポートするスクリプト。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from nyoka import skl_to_pmml
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

SEX_MAP = {"male": 1.0, "female": 0.0}
EMBARKED_MAP = {"S": 0.0, "C": 1.0, "Q": 2.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic RandomForest and export ONNX.")
    parser.add_argument(
        "--csv",
        default="data/Titanic-Dataset.csv",
        help="学習データのCSVパス (default: data/Titanic-Dataset.csv)",
    )
    parser.add_argument(
        "--output",
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


def load_and_preprocess(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[["Survived", *FEATURE_COLUMNS]].copy()

    df["Sex"] = (
        df["Sex"].str.lower().map(SEX_MAP)
    )
    df["Embarked"] = df["Embarked"].map(EMBARKED_MAP)

    # 欠損値を埋める（代入式にしてチェーン警告を回避）
    df["Sex"] = df["Sex"].fillna(0.0)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode(dropna=True)[0])
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLUMNS].astype(np.float32).to_numpy()
    y = df["Survived"].astype(np.int64).to_numpy()
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    return clf


def export_onnx(model: RandomForestClassifier, feature_count: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("float_input", FloatTensorType([None, feature_count]))]
    options = {"zipmap": False}  # 出力をTensor形式にする
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
    output_path.write_bytes(onnx_model.SerializeToString())
    print(f"Saved ONNX model to {output_path}")


def export_pmml(model: RandomForestClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # NyokaはPipelineインスタンスを要求するので単一推定器でもPipelineに包む
    pipeline = Pipeline([("estimator", model)])
    skl_to_pmml(
        pipeline=pipeline,
        col_names=FEATURE_COLUMNS,
        target_name="Survived",
        pmml_f_name=str(output_path),
    )
    print(f"Saved PMML model to {output_path}")


def main() -> None:
    args = parse_args()
    X, y = load_and_preprocess(args.csv)
    model = train_model(X, y, args.estimators)
    export_onnx(model, X.shape[1], Path(args.output))
    export_pmml(model, Path(args.pmml))
    print("Feature order for inference:")
    for idx, name in enumerate(FEATURE_COLUMNS):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
