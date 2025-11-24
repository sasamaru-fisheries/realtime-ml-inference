package com.example.onnx; // onnxパッケージに属するクラスであることを示す

import ai.onnxruntime.OrtException; // ONNX Runtime関連の例外を扱うためのインポート
import org.yaml.snakeyaml.Yaml; // スキーマYAMLを読み込むためのSnakeYAMLインポート

import java.io.IOException; // ファイル入出力でIOExceptionを扱うためのインポート
import java.nio.file.Files; // ファイル操作ユーティリティのインポート
import java.nio.file.Path; // ファイルパスを表すPathクラスのインポート
import java.util.*; // ListやMapなど標準コレクションをまとめてインポート

/**
 * 列名付きの入力をそのままONNXに渡すCSVバッチ推論用ランナー。 // このクラスの概要を説明するコメント
 * スキーマ(YAML)に numeric / categorical の列名を指定し、前処理はモデル側（ONNX内のOneHot/Imputer等）に任せる。 // 前処理をモデル側で行う方針を記載
 *
 * 使い方: // 実行例の説明開始
 *   ModelRunner <model.onnx> <output-name> --csv <csv-path> <schema.yaml> [output-csv] // CLIの基本的な使い方を示す
 *   例) ModelRunner model.onnx probabilities --csv data.csv schema.yaml preds.csv // 使用例として引数を例示
 */
public final class ModelRunner { // インスタンス化させないユーティリティ的な最終クラス

    public static void main(String[] args) { // エントリーポイントのmainメソッド
        if (args.length < 3) { // 引数数が不足していないかチェック
            printUsageAndExit(); // 使い方を表示して終了する
        }
        String modelPath = args[0]; // 最初の引数をONNXモデルパスとして受け取る
        String outputName = "probabilities"; // 出力ノード名のデフォルト値を設定
        int csvArgIndex; // --csv位置を保持するための変数
        if ("--csv".equals(args[1])) { // 第2引数が--csvかどうかを判定
            csvArgIndex = 1; // 出力名が省略された場合の--csv位置
        } else { // 出力名が明示的に指定された場合
            outputName = args[1]; // 指定された出力名を利用
            csvArgIndex = 2; // --csvの位置を後ろにずらす
        }
        if (args.length < csvArgIndex + 3 || !"--csv".equals(args[csvArgIndex])) { // --csvと必要引数が揃っているか確認
            printUsageAndExit(); // 不足していれば使い方を表示して終了
        }
        String csvPath = args[csvArgIndex + 1]; // CSVファイルパスを取得
        String schemaPath = args[csvArgIndex + 2]; // スキーマYAMLパスを取得
        String outputCsv = args.length > csvArgIndex + 3 ? args[csvArgIndex + 3] : null; // 追加で出力CSVがあれば取得、無ければnull

        runCsvMode(modelPath, outputName, csvPath, schemaPath, outputCsv); // CSVモードで推論を実行
    }

    private static void runCsvMode(String modelPath, // モデルパスを受け取る引数
                                   String outputName, // 出力ノード名を受け取る引数
                                   String csvPath, // 入力CSVパスを受け取る引数
                                   String schemaPath, // スキーマYAMLパスを受け取る引数
                                   String outputCsv) { // 推論結果を書き出すCSVパスを受け取る引数
        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) { // 推論器をtry-with-resourcesで準備し自動クローズ
            Schema schema = Schema.load(Path.of(schemaPath)); // スキーマYAMLを読み込む
            java.nio.file.Path path = java.nio.file.Path.of(csvPath); // CSVパスをPathに変換
            List<String> lines = java.nio.file.Files.readAllLines(path); // CSVファイルを行単位で全読み込み
            if (lines.isEmpty()) { // CSVが空かどうかチェック
                System.err.println("CSVが空です: " + csvPath); // 空ならエラーメッセージを出力
                return; // 処理を終了
            }

            String[] headers = lines.get(0).split(","); // 先頭行をヘッダーとしてカンマ分割
            Map<String, Integer> indexByName = new HashMap<>(); // 列名からインデックスを引くマップを用意
            for (int i = 0; i < headers.length; i++) { // ヘッダー配列を走査
                indexByName.put(headers[i].trim().toLowerCase(), i); // 小文字化してマップに登録
            }

            String[] columns = schema.columns(); // スキーマに定義された列名を取得
            List<String> outputLines = new ArrayList<>(); // 推論結果を書き出す行リスト
            boolean headerWritten = false; // 結果ヘッダーを書いたかを追跡
            int row = 0; // 処理したデータ行数のカウンタ
            for (int lineIdx = 1; lineIdx < lines.size(); lineIdx++) { // 2行目以降をデータとして処理
                String line = lines.get(lineIdx); // 現在の行文字列を取得
                if (line.isBlank()) continue; // 空行はスキップ
                String[] parts = line.split(","); // 行をカンマ分割して各列値を得る
                Map<String, Object> inputMap = new HashMap<>(); // 推論入力の列名→値マップを用意
                for (String col : columns) { // スキーマの列ごとに処理
                    String keyLower = col.toLowerCase(); // 列名を小文字化
                    Integer idx = indexByName.get(keyLower); // ヘッダーから対応インデックスを取得
                    if (idx == null || idx >= parts.length) continue; // インデックスが無い場合はスキップ
                    String raw = parts[idx].trim(); // 対応するセル値をトリムして取得
                    if (schema.isNumeric(col)) { // 数値列かどうか判定
                        inputMap.put(col, new float[]{parseFloatSafe(raw)}); // floatとしてパースし配列で格納
                    } else { // カテゴリ列の場合
                        inputMap.put(col, new String[]{raw}); // 文字列のまま配列で格納
                    }
                }
                InferenceResult result = predictor.runInference(inputMap, outputName); // ONNX推論を実行
                float[] out = result.output(); // 推論出力を取得
                if (!headerWritten) { // 最初の結果時にヘッダー行を作成
                    outputLines.add("row" + buildHeader(out.length) + ",time_ms,time_ns"); // 行番号と確率列、計測時間のヘッダーを設定
                    headerWritten = true; // ヘッダーを書いたことを記録
                }
                System.out.printf("Row %d -> %s | time: %.3f ms (%d ns)%n", // コンソールに推論結果と時間を表示
                        lineIdx, Arrays.toString(out), result.elapsedMillis(), result.elapsedNanos()); // printfの書式に合わせた引数を渡す
                outputLines.add(buildLine(lineIdx, out, result)); // 出力用リストにCSV行を追加
                row++; // 処理済み行数をインクリメント
            }
            if (outputCsv != null && outputLines.size() > 1) { // 出力CSV指定があり結果がある場合
                java.nio.file.Path outPath = java.nio.file.Path.of(outputCsv); // 出力パスをPathに変換
                java.nio.file.Files.createDirectories(outPath.getParent() == null ? java.nio.file.Path.of(".") : outPath.getParent()); // 親ディレクトリを事前に作成
                java.nio.file.Files.write(outPath, outputLines); // 結果行を書き出す
                System.out.println("Saved predictions to " + outPath.toAbsolutePath()); // 保存先を案内
            }
        } catch (Exception ex) { // 読み込みや推論で発生した例外をまとめて捕捉
            System.err.println("Failed to run CSV inference: " + ex.getMessage()); // 失敗メッセージを標準エラーに出力
            ex.printStackTrace(System.err); // スタックトレースを標準エラーに出力
            System.exit(2); // 異常終了コードでプロセスを終了
        }
    }

    private static float parseFloatSafe(String raw) { // 数値パースを安全に行うユーティリティ
        try { // パースを試みる
            return Float.parseFloat(raw); // 成功すればそのまま返す
        } catch (NumberFormatException ex) { // 数値に変換できない場合
            return 0f; // 0にフォールバックして返す
        }
    }

    private static String buildHeader(int classes) { // 結果ヘッダーを作るメソッド
        StringBuilder sb = new StringBuilder(); // 可変文字列バッファを生成
        for (int c = 0; c < classes; c++) { // クラス数分ループ
            sb.append(",prob_").append(c); // 各クラスの確率列名を連結
        }
        return sb.toString(); // 完成したヘッダー文字列を返す
    }

    private static String buildLine(int row, float[] preds, InferenceResult result) { // 推論結果1行分をCSV形式に整形
        StringBuilder sb = new StringBuilder(); // 出力用バッファを生成
        sb.append(row); // 行番号を追加
        for (float p : preds) { // 予測値を順に処理
            sb.append(",").append(p); // カンマ区切りで予測値を追加
        }
        sb.append(",").append(String.format("%.3f", result.elapsedMillis())); // 計測ミリ秒を文字列で追加
        sb.append(",").append(result.elapsedNanos()); // 計測ナノ秒を追加
        return sb.toString(); // 完成したCSV行を返す
    }

    private static void printUsageAndExit() { // 使い方を表示して終了するヘルパー
        // 標準エラーに複数行文字列でヘルプを出力
        System.err.println("""
                使い方:
                  CSV一括推論（前処理はモデルに任せる）:
                     ModelRunner <model.onnx> [output-name] --csv <csv-path> <schema.yaml> [output-csv]
                       output-name: 省略時は probabilities を使用
                       schema.yaml: numeric/categorical の列名を持つYAML（train側と合わせる）
                       output-csv: 指定すると推論結果をCSVに保存
                       例) ModelRunner model.onnx probabilities --csv data.csv schema.yaml preds.csv
                           ModelRunner model.onnx --csv data.csv schema.yaml
                """); // ここまでがヘルプメッセージ
        System.exit(1); // 正常系とは違う終了コードで終了
    }

    private record Schema(List<String> numeric, List<String> categorical) { // 数値列とカテゴリ列を保持するレコード
        static Schema load(Path schemaPath) throws IOException { // YAMLからスキーマを読み込む静的メソッド
            Yaml yaml = new Yaml(); // SnakeYAMLインスタンスを生成
            Map<String, Object> map = yaml.load(Files.readString(schemaPath)); // YAMLをMapに読み込む
            List<String> num = extract(map.get("numeric")); // numeric項目を抽出
            List<String> cat = extract(map.get("categorical")); // categorical項目を抽出
            return new Schema(num, cat); // レコードを作成して返す
        }

        private static List<String> extract(Object obj) { // YAMLのエントリをStringリストに変換するヘルパー
            if (obj == null) return List.of(); // 値が無ければ空リストを返す
            List<?> raw = (List<?>) obj; // 未型付けリストとして受け取る
            List<String> out = new ArrayList<>(); // 出力用リストを用意
            for (Object o : raw) { // 要素を順に処理
                if (o instanceof Map<?,?> m && m.containsKey("name")) { // nameキーを持つMap形式の場合
                    out.add(String.valueOf(m.get("name"))); // name値を文字列として追加
                } else { // それ以外の形式の場合
                    out.add(String.valueOf(o)); // そのまま文字列化して追加
                }
            }
            return out; // 変換後のリストを返す
        }

        boolean isNumeric(String col) { // 列がnumericに含まれるか判定するメソッド
            return numeric.stream().anyMatch(c -> c.equalsIgnoreCase(col)); // 大文字小文字を無視して一致を確認
        }

        String[] columns() { // numericとcategoricalを結合して配列化するメソッド
            List<String> all = new ArrayList<>(); // 結合用リストを生成
            all.addAll(numeric); // 数値列を追加
            all.addAll(categorical); // カテゴリ列を追加
            return all.toArray(new String[0]); // 配列に変換して返す
        }
    }
} // ModelRunnerクラスの終端
