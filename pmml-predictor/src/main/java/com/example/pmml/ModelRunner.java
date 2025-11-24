package com.example.pmml; // pmmlパッケージに属するランナークラス

import org.apache.commons.csv.CSVFormat; // CSVフォーマット定義を扱うクラスをインポート
import org.apache.commons.csv.CSVParser; // CSVパースを行うクラスをインポート
import org.apache.commons.csv.CSVRecord; // CSVの1レコードを表すクラスをインポート
import org.yaml.snakeyaml.Yaml; // スキーマYAML読み込み用のライブラリをインポート

import java.io.IOException; // 入出力例外を扱うインポート
import java.io.Reader; // 読み取り用Readerを扱うインポート
import java.nio.file.Files; // ファイル操作ユーティリティのインポート
import java.nio.file.Path; // ファイルパスを表すPathクラスをインポート
import java.util.ArrayList; // 動的配列リストをインポート
import java.util.HashMap; // ハッシュマップ実装をインポート
import java.util.List; // Listインターフェースをインポート
import java.util.Map; // Mapインターフェースをインポート

/**
 * スキーマ(YAML)を読み込み、前処理はモデル側に任せて推論するPMMLランナー。 // 本クラスの役割を説明
 *
 * 使い方: // 実行方法の紹介
 *   ModelRunner <pmml-path> --csv <csv-path> <schema.yaml> [output-csv] // 引数の並びを説明
 *   例) ModelRunner model.pmml --csv data.csv schema.yaml preds.csv // 実行例の提示
 */
public final class ModelRunner { // PMML推論を実行するユーティリティクラス

    public static void main(String[] args) { // エントリーポイントのmainメソッド
        if (args.length < 4 || !"--csv".equals(args[1])) { // 必須引数が揃っているか確認
            printUsageAndExit(); // 不足時は使い方を出して終了
        }
        String pmmlPath = args[0]; // PMMLモデルパスを取得
        String csvPath = args[2]; // 入力CSVパスを取得
        String schemaPath = args[3]; // スキーマYAMLパスを取得
        String outputCsv = args.length >= 5 ? args[4] : null; // 出力CSVが指定されていれば取得

        runCsv(pmmlPath, csvPath, schemaPath, outputCsv); // CSV推論を実行
    }

    private static void runCsv(String pmmlPath, String csvPath, String schemaPath, String outputCsv) { // CSVを一括で推論するメソッド
        try (PmmlPredictor predictor = new PmmlPredictor(pmmlPath)) { // PMML推論器を用意（自動クローズ）
            Schema schema = Schema.load(Path.of(schemaPath)); // スキーマを読み込む
            Path path = Path.of(csvPath); // CSVパスをPath化
            Reader reader = Files.newBufferedReader(path); // CSVファイル用のReaderを開く
            CSVParser parser = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader); // ヘッダー付きでCSVをパース
            Map<String, Integer> indexByName = new HashMap<>(); // 列名→インデックスのマップを作成
            parser.getHeaderMap().forEach((k, v) -> indexByName.put(k.trim().toLowerCase(), v)); // ヘッダーを小文字化して登録

            List<String> outLines = new ArrayList<>(); // 出力CSVの行を保持するリスト
            outLines.add("row,prob_0,prob_1,time_ms,time_ns"); // ヘッダー行を追加

            int row = 0; // 処理行数カウンタ
            int lineIdx = 1; // CSV上の行番号（ヘッダーを除く）
            for (CSVRecord record : parser) { // CSVレコードを順に処理
                Map<String, Object> inputMap = new HashMap<>(); // 推論入力マップを用意
                for (String col : schema.columns()) { // スキーマ列を走査
                    String key = col.toLowerCase(); // 列名を小文字化
                    Integer idx = indexByName.get(key); // インデックスを取得
                    if (idx == null) { // 見つからない場合
                        throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません"); // 必須列不足を通知
                    }
                    String raw = record.get(idx).trim(); // 該当セルを取得してトリム
                    if (schema.isNumeric(col)) { // 数値列かどうか判定
                        inputMap.put(col, parseFloatSafe(raw)); // 数値に変換して格納
                    } else { // カテゴリ列の場合
                        inputMap.put(col, raw); // 文字列のまま格納
                    }
                }
                InferenceResult result = predictor.runInference(inputMap); // PMMLで推論を実行
                Map<String, Double> probs = result.probabilities(); // 出力確率マップを取得
                double prob0 = probs.getOrDefault("0", 0.0); // クラス0の確率を取得（無ければ0）
                double prob1 = probs.getOrDefault("1", 0.0); // クラス1の確率を取得（無ければ0）
                outLines.add((lineIdx) + "," + prob0 + "," + prob1 // 行番号と確率をCSV形式で追加
                        + "," + String.format("%.3f", result.elapsedMillis()) + "," + result.elapsedNanos()); // 計測時間も付与
                System.out.printf("Row %d -> prob0=%.4f, prob1=%.4f, time=%.3f ms%n", // コンソールに結果を表示
                        lineIdx, prob0, prob1, result.elapsedMillis()); // printfの引数を設定
                row++; // 処理行数をインクリメント
                lineIdx++; // CSV上の行番号を進める
            }
            if (outputCsv != null) { // 出力CSVパスが指定されている場合
                Path outPath = Path.of(outputCsv); // 出力パスをPath化
                Files.createDirectories(outPath.getParent() == null ? Path.of(".") : outPath.getParent()); // 親ディレクトリを作成
                Files.write(outPath, outLines); // 推論結果をCSVに書き出す
                System.out.println("Saved PMML predictions to " + outPath.toAbsolutePath()); // 保存先を表示
            }
            System.out.println("Processed rows: " + row); // 処理した行数を表示
        } catch (Exception ex) { // 例外をまとめて捕捉
            System.err.println("Failed to run PMML CSV inference: " + ex.getMessage()); // エラーメッセージを出力
            ex.printStackTrace(System.err); // スタックトレースを出力
            System.exit(2); // 異常終了コードで終了
        }
    }

    private static float parseFloatSafe(String raw) { // 数値変換を安全に行うヘルパー
        try { // パースを試みる
            return Float.parseFloat(raw); // 成功時はfloat値を返す
        } catch (NumberFormatException ex) { // パース失敗時
            return 0f; // 0にフォールバック
        }
    }

    private static void printUsageAndExit() { // 使い方を表示して終了するヘルパー
        System.err.println(""" // 標準エラーにヘルプメッセージを出力
                使い方:
                  CSV一括推論（前処理はモデル側）:
                     ModelRunner <pmml-path> --csv <csv-path> <schema.yaml> [output-csv]
                       schema.yaml: numeric/categorical の列名を持つYAML（train側と合わせる）
                       output-csv: 指定すると予測結果をCSVに保存
                       例) ModelRunner model.pmml --csv data.csv schema.yaml preds.csv
                """); // メッセージの終端
        System.exit(1); // 使い方表示後に終了
    }

    private record Schema(List<String> numeric, List<String> categorical) { // スキーマ情報を保持するレコード
        static Schema load(Path schemaPath) throws IOException { // YAMLからスキーマを読み込む静的メソッド
            Yaml yaml = new Yaml(); // SnakeYAMLインスタンスを生成
            Map<String, Object> map = yaml.load(Files.readString(schemaPath)); // YAMLをMapとして読み込む
            List<String> num = extract(map.get("numeric")); // numericセクションを抽出
            List<String> cat = extract(map.get("categorical")); // categoricalセクションを抽出
            return new Schema(num, cat); // スキーマレコードを生成して返す
        }

        private static List<String> extract(Object obj) { // YAML要素を文字列リストに変換するヘルパー
            if (obj == null) return List.of(); // 要素が無ければ空リスト
            List<?> raw = (List<?>) obj; // 汎用リストとして受け取る
            List<String> out = new ArrayList<>(); // 出力リストを用意
            for (Object o : raw) { // 各要素を処理
                if (o instanceof Map<?, ?> m && m.containsKey("name")) { // nameキーを持つマップ形式の場合
                    out.add(String.valueOf(m.get("name"))); // name値を取り出して追加
                } else { // それ以外の場合
                    out.add(String.valueOf(o)); // 文字列化して追加
                }
            }
            return out; // 結果リストを返す
        }

        boolean isNumeric(String col) { // 列がnumericに含まれるか判定
            return numeric.stream().anyMatch(c -> c.equalsIgnoreCase(col)); // 大文字小文字を無視して一致確認
        }

        String[] columns() { // numericとcategoricalを結合して配列化
            List<String> all = new ArrayList<>(); // すべての列をまとめるリスト
            all.addAll(numeric); // 数値列を追加
            all.addAll(categorical); // カテゴリ列を追加
            return all.toArray(new String[0]); // 配列として返す
        }
    }
} // ModelRunnerクラスの終端
