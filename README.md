## Java ONNX Predictor

`onnx-predictor` ディレクトリには、Pythonで作成したONNXモデルをJavaから読み込み推論するための簡易Mavenプロジェクトが入っています。Titanicデータセットで学習したランダムフォレストをONNX化し、Javaで推論するまでの手順は以下の通りです。

### 1. Pythonでモデルを学習してONNXを書き出す

事前に必要パッケージをインストールします。

```bash
pip install pandas numpy scikit-learn skl2onnx onnx sklearn2pmml pyyaml
```

TitanicのCSV（`data/Titanic-Dataset.csv`）を使って学習＋ONNX/PMMLエクスポート:

```bash
uv run src/train_random_forest.py --csv data/Titanic-Dataset.csv --onnx models/titanic_random_forest.onnx --pmml models/titanic_random_forest.pmml
```

スキーマをカスタムしたい場合は、CSVを参照して推定したテンプレートを出力し、必要な列だけ上書きしてください:

```bash
uv run python src/generate_schema.py --csv data/Titanic-Dataset.csv --target Survived --output schema.yaml
# schema.yaml を編集したのち（--schema は必須）
uv run python src/train_random_forest.py --csv data/Titanic-Dataset.csv --schema schema.yaml --onnx models/titanic_random_forest.onnx --pmml models/titanic_random_forest.pmml
```

実行が成功すると `models/titanic_random_forest.onnx` と `models/titanic_random_forest.pmml` が生成され、標準出力に特徴量の並び順が表示されます。

### 2. Javaプロジェクトのビルド

```bash
cd onnx-predictor
mvn package
```

成功すると通常のJARに加えて依存込みの `target/onnx-predictor-1.0.0-SNAPSHOT-shaded.jar` が生成されます。以降はこのshaded JARを使うと依存クラスパスを個別指定する必要がありません。

### 3. Javaで推論を実行（前処理はモデル側）

`ModelRunner` はスキーマYAMLを読み、CSVを列名付きでモデルに渡します（OneHot/ImputerはONNX内に含まれています）。

#### ONNX: CSV一括推論

```bash
java -jar target/onnx-predictor-1.0.0.jar ../models/titanic_random_forest.onnx probabilities --csv ../data/Titanic-Dataset.csv ../schema.yaml ../models/onnx_predictions.csv
```

- `probabilities` はONNXの出力名（モデルに合わせて変更可）。
- `schema.yaml` は numeric/categorical 列を定義したYAML（`generate_schema.py` で作成）。列名はONNX出力時のスキーマと一致させてください。
- 最後の引数（出力CSV）は省略可。指定しない場合は標準出力のみ。

### データセットを変えたときに直す場所
- ONNX推論: 列名やカテゴリ列を変えたら `schema.yaml` を新データに合わせて更新し、モデルを再生成してください。推論時は `ModelRunner <model.onnx> <output-name> --csv <csv> <schema.yaml> [out.csv]` を使います。
- PMML推論: 同様に `schema.yaml` を更新し、PMMLを新データで再生成してください。PMMLの `ModelRunner` は `--csv <csv> <schema.yaml>` を期待します。
- Python側: 新しいデータ用に `train_random_forest.py` を実行し、ONNX/PMMLを再出力。スキーマを使う場合は `schema.yaml` を更新してください。

## PMML Predictor

`pmml-predictor` ディレクトリには、上記で生成したPMMLファイルを読み込み推論するためのMavenプロジェクトが入っています。

### ビルド

```bash
cd pmml-predictor
mvn package
```

成功すると `target/pmml-predictor-1.0.0.jar` が生成されます（依存込みのshaded JAR）。

### 推論の実行例

PMMLモデルを使って推論する例です。列定義はスキーマYAMLに従います。

```bash
java -jar target/pmml-predictor-1.0.0.jar ../models/titanic_random_forest.pmml --csv ../data/Titanic-Dataset.csv ../schema.yaml ../models/pmml_predictions.csv
```

- `--csv` の後にCSVパス、その次にスキーマYAMLへのパスを指定します（学習時と同じもの）。
- 最後の引数（出力CSV）は省略可。指定しない場合は標準出力のみ。
- 実行結果として確率と推論時間が表示され、CSV出力にも推論時間が含まれます。
