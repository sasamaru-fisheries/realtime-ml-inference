## Java ONNX Predictor

`onnx-predictor` ディレクトリには、Pythonで作成したONNXモデルをJavaから読み込み推論するための簡易Mavenプロジェクトが入っています。Titanicデータセットで学習したランダムフォレストをONNX化し、Javaで推論するまでの手順は以下の通りです。

### 1. Pythonでモデルを学習してONNXを書き出す

事前に必要パッケージをインストールします。

```bash
pip install pandas numpy scikit-learn skl2onnx onnx nyoka
```

TitanicのCSV（`data/Titanic-Dataset.csv`）を使って学習＋ONNX/PMMLエクスポート:

```bash
python train_random_forest.py \
  --csv data/Titanic-Dataset.csv \
  --output models/titanic_random_forest.onnx \
  --pmml models/titanic_random_forest.pmml
```

実行が成功すると `models/titanic_random_forest.onnx` と `models/titanic_random_forest.pmml` が生成され、標準出力に特徴量の並び順が表示されます（`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`）。この順番で数値を並べて推論用の入力を作成してください。

### 2. Javaプロジェクトのビルド

```bash
cd onnx-predictor
mvn package
```

成功すると通常のJARに加えて依存込みの `target/onnx-predictor-1.0.0-SNAPSHOT-shaded.jar` が生成されます。以降はこのshaded JARを使うと依存クラスパスを個別指定する必要がありません。

### 3. Javaで推論を実行

`ModelRunner` の引数には「モデルパス」「入力テンソル名」「出力テンソル名」「カンマ区切りの入力値」「shape(optional)」を渡します。今回の学習スクリプトで生成されるONNXは `float_input` を受け取り、出力名は `label`（予測ラベル）と `probabilities`（各クラス確率）の2種類です。

```bash
java -jar target/onnx-predictor-1.0.0-SNAPSHOT-shaded.jar \
    ../models/titanic_random_forest.onnx \
    float_input \
    probabilities \
    3,1,29,0,0,7.25,0 \
    1,7
```

- 5番目の引数はPython側で表示された特徴量順に並べた入力値です。
- 6番目の引数（shape）は省略可能で、省略時は `1,入力値の個数` が自動で使われます。
- `probabilities` を指定すると `[生存しない確率, 生存する確率]` が返ります。ラベルが欲しい場合は `label` を指定してください（この場合はlong配列が返る点に注意）。
- 推論実行時には `Inference time: ... ms` が出力され、1回の推論に要した時間を確認できます。

#### CSVを使った一括推論

CSVから特徴量を読み込んで複数行まとめて推論する場合:

```bash
java -jar target/onnx-predictor-1.0.0-SNAPSHOT-shaded.jar \
    ../models/titanic_random_forest.onnx \
    float_input \
    probabilities \
    --csv data/Titanic-Dataset.csv \
    Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
```

- `--csv` の後にCSVパス、その後にモデルが期待する列名をカンマ区切りで順序指定します。
- 各行について推論結果と推論時間が表示されます。

## PMML Predictor

`pmml-predictor` ディレクトリには、上記で生成したPMMLファイルを読み込み推論するためのMavenプロジェクトが入っています。

### ビルド

```bash
cd pmml-predictor
mvn package
```

成功すると `target/pmml-predictor-1.0.0.jar` が生成されます（依存込みのshaded JAR）。

### 推論の実行例

Titanicモデル（pmml）を使って推論する場合の例です。入力値の順序は `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked` です。

```bash
java -jar target/pmml-predictor-1.0.0.jar \
    ../models/titanic_random_forest.pmml \
    3,1,29,0,0,7.25,0 \
    Survived
```

- 2番目の引数はカンマ区切りの入力値です。
- 3番目の引数は予測ターゲットのフィールド名です（Titanicでは `Survived`）。
- 実行結果として予測ラベルと確率分布、推論時間が表示されます。
