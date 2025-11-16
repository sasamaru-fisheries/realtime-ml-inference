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
from sklearn.impute import SimpleImputer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from nyoka import skl_to_pmml

NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_COLUMNS = ["Sex", "Embarked"]
FEATURE_COLUMNS = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]

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

    # パイプラインに前処理を組み込むため、ここでは特徴とラベルを切り出すのみ
    X = df[FEATURE_COLUMNS]
    y = df["Survived"].astype(np.int64).to_numpy()
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
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


def export_onnx(model: RandomForestClassifier, feature_count: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = []
    for col in NUMERIC_COLUMNS:
        initial_type.append((col, FloatTensorType([None, 1])))
    for col in CATEGORICAL_COLUMNS:
        initial_type.append((col, StringTensorType([None, 1])))
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
