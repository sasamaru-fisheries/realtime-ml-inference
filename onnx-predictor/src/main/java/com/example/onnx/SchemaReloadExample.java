package com.example.onnx; // onnxパッケージのサンプルクラス

import org.yaml.snakeyaml.Yaml; // スキーマYAMLを読み込むためのライブラリをインポート

import java.io.IOException; // 入出力例外を扱うためのインポート
import java.nio.file.Files; // ファイル読み込みユーティリティのインポート
import java.nio.file.Path; // ファイルパスを表すPathクラスのインポート
import java.util.*; // ListやMapなどコレクションをまとめてインポート

/**
 * スキーマ(YAML)を読み込み、CSVの任意行（またはデフォルト値）で // このサンプルの概要説明
 * 複数モデルを同一JVM内で順に推論するサンプル。 // 2つのモデルを切り替えて推論することを示す
 *
 * 使い方: // 実行方法の説明
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SchemaReloadExample \ // クラスパス付きの実行例を示す
 *     <model1.onnx> <model2.onnx> <schema.yaml> [csv] [rowIndex(1-based)] [output-name] // 引数の並びを説明
 *
 * 例: // 具体例の開始
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SchemaReloadExample \ // 例の実行コマンド
 *     ../models/model1.onnx ../models/model2.onnx ../schema.yaml ../data/input.csv 2 probabilities // 例に使うファイルと引数
 */
public final class SchemaReloadExample { // サンプルをまとめたユーティリティクラス
    public static void main(String[] args) throws Exception { // エントリーポイントのmainメソッド
        if (args.length < 3) { // 必須引数が足りない場合をチェック
            // 標準エラーに使い方を表示
            System.err.println("""
                    使い方:
                      SchemaReloadExample <model1.onnx> <model2.onnx> <schema.yaml> [csv] [rowIndex] [output-name]
                        csv: (任意) 入力データのCSV。指定しない場合はスキーマに従ってデフォルト(0/空文字)を使用
                        rowIndex: (任意) CSVの何行目を使うか（1始まり、ヘッダーを含めない）
                        output-name: (任意) ONNX出力名。省略時は probabilities
                    """); // ヘルプメッセージ終端
            System.exit(1); // 使い方表示後に終了
        }
        String model1 = args[0]; // 1つ目のモデルパスを取得
        String model2 = args[1]; // 2つ目のモデルパスを取得
        String schemaPath = args[2]; // スキーマYAMLパスを取得
        String csvPath = args.length >= 4 ? args[3] : null; // 任意のCSVパスを取得（省略可能）
        int rowIndex = args.length >= 5 ? Integer.parseInt(args[4]) : 1; // 読み込む行番号を取得、指定が無ければ1
        String outputName = args.length >= 6 ? args[5] : "probabilities"; // 出力ノード名を取得、無ければデフォルト

        Schema schema = Schema.load(Path.of(schemaPath)); // スキーマをロード
        Map<String, Object> input = csvPath != null // CSV指定の有無で入力の作り方を分岐
                ? loadRowFromCsv(csvPath, schema, rowIndex) // CSVから指定行を読み込む場合
                : buildDefaultInputs(schema); // デフォルトの0/空文字を使う場合

        try (OnnxPredictor predictor = new OnnxPredictor(model1)) { // 1つのPredictorを使い回す
            InferenceResult r1 = predictor.runInference(input, outputName); // モデル1で推論を実行
            System.out.println("Model1 output: " + Arrays.toString(r1.output())); // 出力配列を表示
            System.out.printf("Elapsed: %.3f ms%n", r1.elapsedMillis()); // 実行時間を表示

            predictor.setModelPath(model2); // model2に差し替える
            predictor.reloadModel();        // セッションを再構築する

            InferenceResult r2 = predictor.runInference(input, outputName); // モデル2で推論を実行
            System.out.println("Model2 output: " + Arrays.toString(r2.output())); // 出力配列を表示
            System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis()); // 実行時間を表示
        }
    }

    private static Map<String, Object> buildDefaultInputs(Schema schema) { // スキーマに基づいてデフォルト入力を作成
        Map<String, Object> map = new HashMap<>(); // 入力マップを初期化
        for (String col : schema.numeric()) { // 数値列を走査
            map.put(col, new float[]{0f}); // 数値列には0を設定
        }
        for (String col : schema.categorical()) { // カテゴリ列を走査
            map.put(col, new String[]{""}); // カテゴリ列には空文字を設定
        }
        return map; // 作成したマップを返す
    }

    private static Map<String, Object> loadRowFromCsv(String csvPath, Schema schema, int rowIndex) throws IOException { // CSVから指定行を読み込む
        List<String> lines = Files.readAllLines(Path.of(csvPath)); // CSVを全行読み込む
        if (lines.size() < 2) { // データ行が無い場合
            throw new IllegalArgumentException("CSVが空かヘッダーのみです: " + csvPath); // 不正を通知
        }
        if (rowIndex < 1 || rowIndex >= lines.size()) { // 行番号が範囲外かチェック
            throw new IllegalArgumentException("rowIndex が範囲外です (1.." + (lines.size() - 1) + ")"); // 範囲外を伝える
        }
        String[] headers = lines.get(0).split(","); // ヘッダー行をカンマ分割
        Map<String, Integer> indexByName = new HashMap<>(); // 列名→インデックスのマップを用意
        for (int i = 0; i < headers.length; i++) { // ヘッダーを走査
            indexByName.put(headers[i].trim().toLowerCase(), i); // 小文字化してマップに登録
        }
        String[] parts = lines.get(rowIndex).split(","); // 対象行をカンマ分割

        Map<String, Object> inputMap = new HashMap<>(); // 入力用マップを初期化
        for (String col : schema.columns()) { // スキーマの全列を処理
            String keyLower = col.toLowerCase(); // 列名を小文字化
            Integer idx = indexByName.get(keyLower); // 対応するインデックスを取得
            if (idx == null || idx >= parts.length) { // インデックスが見つからないか範囲外の場合
                throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません"); // 必須列不足を通知
            }
            String raw = parts[idx].trim(); // 対象セルの値をトリム
            if (schema.isNumeric(col)) { // 数値列かどうか判定
                inputMap.put(col, new float[]{parseFloatSafe(raw)}); // 数値としてパースして格納
            } else { // カテゴリ列の場合
                inputMap.put(col, new String[]{raw}); // 文字列のまま格納
            }
        }
        return inputMap; // 作成した入力マップを返す
    }

    private static float parseFloatSafe(String raw) { // 数値変換を安全に行うヘルパー
        try { // パースを試す
            return Float.parseFloat(raw); // 成功時は結果を返す
        } catch (NumberFormatException ex) { // 数値でない場合
            return 0f; // 0にフォールバック
        }
    }

    private record Schema(List<String> numeric, List<String> categorical) { // スキーマ情報を保持するレコード
        static Schema load(Path schemaPath) throws IOException { // YAMLからスキーマを読み込む静的メソッド
            Yaml yaml = new Yaml(); // SnakeYAMLのインスタンス生成
            Map<String, Object> map = yaml.load(Files.readString(schemaPath)); // YAMLをMapに読み込む
            List<String> num = extract(map.get("numeric")); // numericセクションを抽出
            List<String> cat = extract(map.get("categorical")); // categoricalセクションを抽出
            return new Schema(num, cat); // スキーマレコードを生成して返す
        }

        private static List<String> extract(Object obj) { // YAML要素をStringリストにするヘルパー
            if (obj == null) return List.of(); // 要素が無ければ空リスト
            List<?> raw = (List<?>) obj; // 汎用リストとして受け取る
            List<String> out = new ArrayList<>(); // 出力用リストを準備
            for (Object o : raw) { // 各要素を処理
                if (o instanceof Map<?, ?> m && m.containsKey("name")) { // nameキーを持つマップ形式の場合
                    out.add(String.valueOf(m.get("name"))); // name値を取り出して追加
                } else { // それ以外の場合
                    out.add(String.valueOf(o)); // 文字列化して追加
                }
            }
            return out; // 抽出結果を返す
        }

        boolean isNumeric(String col) { // 列がnumericに含まれるか調べる
            return numeric.stream().anyMatch(c -> c.equalsIgnoreCase(col)); // 大文字小文字無視で一致を確認
        }

        String[] columns() { // numericとcategoricalを結合した配列を返す
            List<String> all = new ArrayList<>(); // 結合用リストを生成
            all.addAll(numeric); // 数値列を追加
            all.addAll(categorical); // カテゴリ列を追加
            return all.toArray(new String[0]); // 配列にして返す
        }
    }
} // SchemaReloadExampleクラスの終端
