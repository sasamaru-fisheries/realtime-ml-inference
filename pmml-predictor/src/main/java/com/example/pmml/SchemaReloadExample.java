package com.example.pmml; // pmmlパッケージのサンプルクラス

import org.yaml.snakeyaml.Yaml; // スキーマYAMLを読み込むためのライブラリをインポート

import java.io.IOException; // 入出力例外に対応するためのインポート
import java.nio.file.Files; // ファイル読み込みユーティリティをインポート
import java.nio.file.Path; // パス操作用のPathクラスをインポート
import java.util.*; // コレクションAPIをまとめてインポート

/**
 * スキーマYAMLを読み込み、CSVの任意行（またはデフォルト値）を使って // サンプルの概要を説明
 * 複数PMMLモデルを同一JVM内で順に推論するサンプル。 // 2つのPMMLモデルを切り替え推論することを示す
 *
 * 使い方: // 実行方法の説明
 *   java -cp target/pmml-predictor-1.0.0.jar com.example.pmml.SchemaReloadExample \ // クラスパス付きの実行例
 *     <model1.pmml> <model2.pmml> <schema.yaml> [csv] [rowIndex] // 引数の説明
 *
 * 例: // 実行例の提示
 *   java -cp target/pmml-predictor-1.0.0.jar com.example.pmml.SchemaReloadExample \ // コマンド例
 *     ../models/model1.pmml ../models/model2.pmml ../schema.yaml ../data/input.csv 2 // モデルと入力例
 */
public final class SchemaReloadExample { // PMMLモデル差し替えサンプルの最終クラス
    public static void main(String[] args) throws Exception { // エントリーポイント
        if (args.length < 3) { // 必須引数が揃っているか確認
            // 標準エラーに使い方を出力
            System.err.println("""
                    使い方:
                      SchemaReloadExample <model1.pmml> <model2.pmml> <schema.yaml> [csv] [rowIndex]
                        csv: (任意) 入力データのCSV。指定しない場合はスキーマに従ってデフォルト(0/空文字)を使用
                        rowIndex: (任意) CSVの何行目を使うか（1始まり、ヘッダーを含めない）
                """); // メッセージ終端
            System.exit(1); // 不足時は終了
        }
        String model1 = args[0]; // 1つ目のPMMLモデルパス
        String model2 = args[1]; // 2つ目のPMMLモデルパス
        String schemaPath = args[2]; // スキーマYAMLパス
        String csvPath = args.length >= 4 ? args[3] : null; // 任意のCSVパス
        int rowIndex = args.length >= 5 ? Integer.parseInt(args[4]) : 1; // CSVの使用行番号、無指定なら1

        Schema schema = Schema.load(Path.of(schemaPath)); // スキーマをロード
        Map<String, Object> input = csvPath != null // CSV指定の有無で入力生成を分岐
                ? loadRowFromCsv(csvPath, schema, rowIndex) // CSVから指定行を読み込む
                : buildDefaultInputs(schema); // 0/空文字のデフォルト入力を作成

        try (PmmlPredictor predictor = new PmmlPredictor(model1)) { // 1インスタンスでモデルを差し替えながら使用
            InferenceResult r1 = predictor.runInference(input); // モデル1で推論
            System.out.println("Model1 probabilities: " + r1.probabilities()); // 確率分布を表示
            System.out.printf("Elapsed: %.3f ms%n", r1.elapsedMillis()); // 実行時間を表示

            predictor.setModelPath(model2); // 2つ目のPMMLへ切り替え
            predictor.reloadModel();        // Evaluatorを再構築

            InferenceResult r2 = predictor.runInference(input); // モデル2で推論
            System.out.println("Model2 probabilities: " + r2.probabilities()); // 確率分布を表示
            System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis()); // 実行時間を表示
        }
    }

    private static Map<String, Object> buildDefaultInputs(Schema schema) { // スキーマに合わせたデフォルト入力を作成
        Map<String, Object> map = new HashMap<>(); // 入力マップを初期化
        for (String col : schema.numeric()) { // 数値列を走査
            map.put(col, 0f); // 数値列には0を設定
        }
        for (String col : schema.categorical()) { // カテゴリ列を走査
            map.put(col, ""); // カテゴリ列には空文字を設定
        }
        return map; // 完成したマップを返す
    }

    private static Map<String, Object> loadRowFromCsv(String csvPath, Schema schema, int rowIndex) throws IOException { // CSVの指定行を読み込む
        List<String> lines = Files.readAllLines(Path.of(csvPath)); // CSVを全行読み込む
        if (lines.size() < 2) { // データ行が存在するかチェック
            throw new IllegalArgumentException("CSVが空かヘッダーのみです: " + csvPath); // 不正を通知
        }
        if (rowIndex < 1 || rowIndex >= lines.size()) { // 行番号が範囲内か確認
            throw new IllegalArgumentException("rowIndex が範囲外です (1.." + (lines.size() - 1) + ")"); // 範囲外エラー
        }
        String[] headers = lines.get(0).split(","); // ヘッダーをカンマ分割
        Map<String, Integer> indexByName = new HashMap<>(); // 列名→インデックスのマップを用意
        for (int i = 0; i < headers.length; i++) { // ヘッダー配列を走査
            indexByName.put(headers[i].trim().toLowerCase(), i); // 小文字化して登録
        }
        String[] parts = lines.get(rowIndex).split(","); // 対象行をカンマ分割

        Map<String, Object> inputMap = new HashMap<>(); // 入力マップを初期化
        for (String col : schema.columns()) { // スキーマの全列を処理
            String keyLower = col.toLowerCase(); // 列名を小文字化
            Integer idx = indexByName.get(keyLower); // 対応インデックスを取得
            if (idx == null || idx >= parts.length) { // インデックスが無い、または範囲外の場合
                throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません"); // 必須列不足を通知
            }
            String raw = parts[idx].trim(); // 対象セルの値をトリム
            if (schema.isNumeric(col)) { // 数値列かどうか判定
                inputMap.put(col, parseFloatSafe(raw)); // 数値としてパースして格納
            } else { // カテゴリ列の場合
                inputMap.put(col, raw); // 文字列のまま格納
            }
        }
        return inputMap; // 作成した入力マップを返す
    }

    private static float parseFloatSafe(String raw) { // 数値変換を安全に行うヘルパー
        try { // パースを試みる
            return Float.parseFloat(raw); // 成功時はfloat値を返す
        } catch (NumberFormatException ex) { // 変換失敗時
            return 0f; // 0にフォールバック
        }
    }

    private record Schema(List<String> numeric, List<String> categorical) { // スキーマ構造を表すレコード
        static Schema load(Path schemaPath) throws IOException { // YAMLからスキーマを読み込む静的メソッド
            Yaml yaml = new Yaml(); // SnakeYAMLインスタンスを生成
            Map<String, Object> map = yaml.load(Files.readString(schemaPath)); // YAMLをMapに読み込む
            List<String> num = extract(map.get("numeric")); // numericセクションを抽出
            List<String> cat = extract(map.get("categorical")); // categoricalセクションを抽出
            return new Schema(num, cat); // レコードを生成して返す
        }

        private static List<String> extract(Object obj) { // YAML要素を文字列リストに変換するヘルパー
            if (obj == null) return List.of(); // 要素が無ければ空リスト
            List<?> raw = (List<?>) obj; // 汎用リストとして扱う
            List<String> out = new ArrayList<>(); // 出力用リストを準備
            for (Object o : raw) { // 要素を走査
                if (o instanceof Map<?, ?> m && m.containsKey("name")) { // nameキー付きマップの場合
                    out.add(String.valueOf(m.get("name"))); // name値を追加
                } else { // それ以外の場合
                    out.add(String.valueOf(o)); // 文字列化して追加
                }
            }
            return out; // 変換結果を返す
        }

        boolean isNumeric(String col) { // 列がnumericに含まれるか判定
            return numeric.stream().anyMatch(c -> c.equalsIgnoreCase(col)); // 大文字小文字無視で一致を確認
        }

        String[] columns() { // numericとcategoricalを結合して配列にする
            List<String> all = new ArrayList<>(); // すべての列をまとめるリスト
            all.addAll(numeric); // 数値列を追加
            all.addAll(categorical); // カテゴリ列を追加
            return all.toArray(new String[0]); // 配列に変換して返す
        }
    }
} // SchemaReloadExampleクラスの終端
