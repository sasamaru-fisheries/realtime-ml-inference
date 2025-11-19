"""TitanicデータでLightGBMを訓練し、ONNX / PMML形式でエクスポートするスクリプト。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from onnxmltools.convert.lightgbm import convert as convert_lightgbm
from skl2onnx import convert_sklearn as convert_preprocess
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType, StringTensorType
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
from onnx.compose import merge_models
from onnx import helper
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic LightGBM and export ONNX / PMML.")
    parser.add_argument(
        "--csv",
        default="data/Titanic-Dataset.csv",
        help="学習データのCSVパス (default: data/Titanic-Dataset.csv)",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="前処理・列定義を含むYAMLスキーマのパス（必須）",
    )
    parser.add_argument(
        "--onnx",
        default="models/titanic_lightgbm.onnx",
        help="出力するONNXファイルパス",
    )
    parser.add_argument(
        "--pmml",
        default="models/titanic_lightgbm.pmml",
        help="出力するPMMLファイルパス",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=300,
        help="LightGBMのn_estimators (default: 300)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBMのlearning_rate (default: 0.05)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="LightGBMのnum_leaves (default: 31)",
    )
    return parser.parse_args()


def load_schema(schema_path: str | None):
    if schema_path is None:
        raise ValueError("schema.yaml は必須です。--schema で指定してください。")
    return yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))


def load_and_preprocess(csv_path: str, schema: dict) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = pd.read_csv(csv_path)
    target = schema["target"]
    numeric_cols = [c["name"] for c in schema.get("numeric", [])]
    categorical_cols = [c["name"] for c in schema.get("categorical", [])]
    feature_cols = numeric_cols + categorical_cols
    df = df[[target, *feature_cols]].copy()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        if df[col].isnull().all():
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    X = df[feature_cols]
    y_series = df[target]
    unique_vals = y_series.dropna().unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"ターゲット列 {target} のクラス数が想定外です: {len(unique_vals)} 個 "
            f"({unique_vals}). 2クラス（0/1など）に揃えてから再学習してください。"
        )
    if set(unique_vals) <= {0, 1}:
        y = y_series.astype(np.int64).to_numpy()
    else:
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        y = y_series.map(mapping).astype(np.int64).to_numpy()
    return X, y, numeric_cols, categorical_cols


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
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
            ("model", lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                objective="binary",
                n_jobs=-1,
                random_state=42,
            )),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    try:
        proba = clf.predict_proba(X_test)
        if proba.shape[1] == 2:
            pos_proba = proba[:, 1]
            roc = roc_auc_score(y_test, pos_proba)
            pr = average_precision_score(y_test, pos_proba)
            print(f"Validation ROC-AUC: {roc:.3f}")
            print(f"Validation PR-AUC:  {pr:.3f}")
        else:
            print(f"predict_proba のクラス数が {proba.shape[1]} です。二値分類のみROC/PR-AUCを計算します。")
    except Exception as ex:
        print(f"ROC/PR-AUC計算に失敗しました: {ex}")
    return clf


def export_onnx(
    model: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
    sample_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = []
    for col in numeric_cols:
        initial_type.append((col, SklearnFloatTensorType([None, 1])))
    for col in categorical_cols:
        initial_type.append((col, StringTensorType([None, 1])))
    preprocess = model.named_steps["preprocess"]
    preprocess_onnx = convert_preprocess(preprocess, initial_types=initial_type, target_opset=15)

    transformed = preprocess.transform(sample_frame.iloc[:1])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_dim = transformed.shape[1]
    lgb_input_name = "preprocessed"
    lgb_initial_type = [(lgb_input_name, OnnxFloatTensorType([None, transformed_dim]))]
    lgb_model = model.named_steps["model"]
    lgb_onnx = convert_lightgbm(lgb_model, initial_types=lgb_initial_type, zipmap=False, target_opset=15)
    max_ir = max(preprocess_onnx.ir_version, lgb_onnx.ir_version)
    preprocess_onnx.ir_version = max_ir
    lgb_onnx.ir_version = max_ir

    def sync_default_opset(model_a, model_b) -> None:
        def get_version(model, domain: str) -> int:
            for opset in model.opset_import:
                if opset.domain == domain:
                    return opset.version
            return 0

        def set_version(model, domain: str, version: int) -> None:
            for opset in model.opset_import:
                if opset.domain == domain:
                    opset.version = version
                    return
            model.opset_import.extend([helper.make_opsetid(domain, version)])

        default_domain_version = max(get_version(model_a, ""), get_version(model_b, ""))
        set_version(model_a, "", default_domain_version)
        set_version(model_b, "", default_domain_version)

    sync_default_opset(preprocess_onnx, lgb_onnx)
    merged_model = merge_models(
        preprocess_onnx,
        lgb_onnx,
        io_map=[(preprocess_onnx.graph.output[0].name, lgb_input_name)],
    )
    output_path.write_bytes(merged_model.SerializeToString())
    print(f"Saved ONNX model to {output_path}")


def export_pmml(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pmml_pipeline = PMMLPipeline(model.steps)
    sklearn2pmml(pmml_pipeline, str(output_path), with_repr=True)
    print(f"Saved PMML model to {output_path}")


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    X, y, numeric_cols, categorical_cols = load_and_preprocess(args.csv, schema)
    model = train_model(
        X,
        y,
        args.estimators,
        args.learning_rate,
        args.num_leaves,
        numeric_cols,
        categorical_cols,
    )
    feature_cols = numeric_cols + categorical_cols
    export_onnx(model, numeric_cols, categorical_cols, X, Path(args.onnx))
    export_pmml(model, Path(args.pmml))
    print("Feature order for inference:")
    for idx, name in enumerate(feature_cols):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
