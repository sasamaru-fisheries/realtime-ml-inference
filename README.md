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

`ModelRunner` の引数には「モデルパス」「入力テンソル名」「出力テンソル名」「入力データ」を渡します。今回の学習スクリプトでは前処理を含んだPipelineをそのままONNX化しているため、CSV経由で列名付きデータを渡すモードを推奨します（`Sex`/`Embarked` が文字列のままでも内部でOneHotEncodeされます）。

#### CSVを使った一括推論

CSVから特徴量を読み込んで複数行まとめて推論する場合:

```bash
java -jar target/onnx-predictor-1.0.0-SNAPSHOT-shaded.jar \
    ../models/titanic_random_forest.onnx \
    float_input \
    output_probability \
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
