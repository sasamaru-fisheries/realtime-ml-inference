package com.example.onnx; // onnxパッケージのサンプル使用クラス

import org.yaml.snakeyaml.Yaml; // スキーマYAMLを扱うためのSnakeYAMLをインポート

import java.io.IOException; // 入出力例外に対処するためのインポート
import java.nio.file.Files; // ファイル読み込みユーティリティをインポート
import java.nio.file.Path; // パス操作のためのPathクラスをインポート
import java.util.*; // コレクションAPIをまとめてインポート

/**
 * スキーマYAMLを読み込み、CSVの任意行（またはデフォルト値）を使って推論するサンプル。 // サンプルの概要を説明
 *
 * 使い方: // 実行方法の説明
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SampleUsage \ // クラスパス付きの実行例
 *     <model.onnx> <output-name|省略でprobabilities> <schema.yaml> [csv-path] [rowIndex(1-based)] // 引数の説明
 *
 * 例: // 具体例の開始
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SampleUsage \ // コマンド例
 *     ../models/titanic_random_forest.onnx probabilities ../schema.yaml ../data/Titanic-Dataset.csv 2 // モデル・スキーマ・CSV例
 */
public final class SampleUsage { // 推論の使い方を示す最終クラス
    public static void main(String[] args) throws Exception { // エントリーポイント
        if (args.length < 2) { // 必須引数が足りない場合のチェック
            // 標準エラーに使い方を表示
            System.err.println("""
                    使い方:
                      SampleUsage <model.onnx> <output-name|省略可> <schema.yaml> [csv] [rowIndex]
                        output-name: 省略時は probabilities
                        csv: (任意) 入力データのCSV。指定しない場合はスキーマで0/空文字のダミーを作成
                        rowIndex: (任意) CSVの何行目を使うか（1始まり、ヘッダーを含めない）
                """); // ヘルプメッセージ終端
            System.exit(1); // 不正利用なので終了
        }

        String modelPath = args[0]; // モデルファイルパス
        String outputName; // 出力ノード名を格納する変数
        String schemaPath; // スキーマYAMLパスを格納する変数
        String csvPath = null; // 任意のCSVパス、初期値はnull
        int rowIndex = 1; // デフォルトの行番号（1始まり）

        if (args.length >= 3) { // 出力名とスキーマが指定されている場合
            outputName = args[1]; // 指定された出力名を設定
            schemaPath = args[2]; // スキーマパスを設定
            if (args.length >= 4) { // CSVパスが指定されている場合
                csvPath = args[3]; // CSVパスを取得
            }
            if (args.length >= 5) { // 行番号が指定されている場合
                rowIndex = Integer.parseInt(args[4]); // 行番号を整数に変換して設定
            }
        } else { // 出力名が省略された場合の分岐
            outputName = "probabilities"; // デフォルト出力名を設定
            schemaPath = args[1]; // 2番目の引数をスキーマパスとして扱う
            if (args.length >= 3) { // CSVパスがある場合
                csvPath = args[2]; // CSVパスを取得
            }
            if (args.length >= 4) { // 行番号がある場合
                rowIndex = Integer.parseInt(args[3]); // 行番号を設定
            }
        }

        Schema schema = Schema.load(Path.of(schemaPath)); // スキーマYAMLを読み込み
        Map<String, Object> inputMap = csvPath != null // CSV指定の有無で入力生成方法を分ける
                ? loadRowFromCsv(csvPath, schema, rowIndex) // CSVから指定行を読み込む
                : buildDefaultInputs(schema); // 0/空文字のダミー入力を生成する

        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) { // 推論器を準備（自動クローズ）
            InferenceResult r = predictor.runInference(inputMap, outputName); // 推論を実行
            System.out.println("Output (" + outputName + "): " + Arrays.toString(r.output())); // 出力内容を表示
            System.out.printf("Elapsed: %.3f ms (%d ns)%n", r.elapsedMillis(), r.elapsedNanos()); // 所要時間を表示
        }
    }

    private static Map<String, Object> buildDefaultInputs(Schema schema) { // スキーマに合わせたデフォルト入力を作成
        Map<String, Object> map = new HashMap<>(); // 入力マップを初期化
        for (String col : schema.numeric()) { // 数値列を走査
            map.put(col, new float[]{0f}); // 数値列には0をセット
        }
        for (String col : schema.categorical()) { // カテゴリ列を走査
            map.put(col, new String[]{""}); // カテゴリ列には空文字をセット
        }
        return map; // 完成したマップを返す
    }

    private static Map<String, Object> loadRowFromCsv(String csvPath, Schema schema, int rowIndex) throws IOException { // CSVの指定行を読み込む
        List<String> lines = Files.readAllLines(Path.of(csvPath)); // CSVを全行読み込む
        if (lines.size() < 2) { // データ行が無い場合のチェック
            throw new IllegalArgumentException("CSVが空かヘッダーのみです: " + csvPath); // 不正を知らせる
        }
        if (rowIndex < 1 || rowIndex >= lines.size()) { // 行番号が範囲外かどうか
            throw new IllegalArgumentException("rowIndex が範囲外です (1.." + (lines.size() - 1) + ")"); // 範囲外を通知
        }
        String[] headers = lines.get(0).split(","); // ヘッダーをカンマ分割
        Map<String, Integer> indexByName = new HashMap<>(); // 列名→インデックスのマップを作成
        for (int i = 0; i < headers.length; i++) { // ヘッダー行を走査
            indexByName.put(headers[i].trim().toLowerCase(), i); // 小文字化してマップに登録
        }
        String[] parts = lines.get(rowIndex).split(","); // 指定行をカンマ分割

        Map<String, Object> inputMap = new HashMap<>(); // 入力マップを初期化
        for (String col : schema.columns()) { // スキーマの全列を処理
            String keyLower = col.toLowerCase(); // 列名を小文字化
            Integer idx = indexByName.get(keyLower); // 対応インデックスを取得
            if (idx == null || idx >= parts.length) { // インデックスが無い、または範囲外の場合
                throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません"); // 必要な列が無いことを通知
            }
            String raw = parts[idx].trim(); // 対象セルの文字列をトリム
            if (schema.isNumeric(col)) { // 数値列かどうか
                inputMap.put(col, new float[]{parseFloatSafe(raw)}); // 数値としてパースして格納
            } else { // カテゴリ列の場合
                inputMap.put(col, new String[]{raw}); // 文字列のまま格納
            }
        }
        return inputMap; // 構築した入力マップを返す
    }

    private static float parseFloatSafe(String raw) { // 数値パースを安全に行うヘルパー
        try { // パースを試みる
            return Float.parseFloat(raw); // 成功時は結果を返す
        } catch (NumberFormatException ex) { // パース失敗時
            return 0f; // 0にフォールバック
        }
    }

    private record Schema(List<String> numeric, List<String> categorical) { // スキーマを保持するレコード
        static Schema load(Path schemaPath) throws IOException { // YAMLからスキーマを読み込む静的メソッド
            Yaml yaml = new Yaml(); // SnakeYAMLインスタンスを生成
            Map<String, Object> map = yaml.load(Files.readString(schemaPath)); // YAMLをMapに読み込む
            List<String> num = extract(map.get("numeric")); // numericセクションを抽出
            List<String> cat = extract(map.get("categorical")); // categoricalセクションを抽出
            return new Schema(num, cat); // レコードを生成して返す
        }

        private static List<String> extract(Object obj) { // YAML要素をStringリストに変換するヘルパー
            if (obj == null) return List.of(); // 要素が無ければ空リスト
            List<?> raw = (List<?>) obj; // 汎用リストとして扱う
            List<String> out = new ArrayList<>(); // 出力リストを準備
            for (Object o : raw) { // 要素を走査
                if (o instanceof Map<?, ?> m && m.containsKey("name")) { // nameキー付きのマップなら
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

        String[] columns() { // numericとcategoricalを結合した配列を返す
            List<String> all = new ArrayList<>(); // 結合用リストを準備
            all.addAll(numeric); // 数値列を追加
            all.addAll(categorical); // カテゴリ列を追加
            return all.toArray(new String[0]); // 配列に変換して返す
        }
    }
} // SampleUsageクラスの終端
