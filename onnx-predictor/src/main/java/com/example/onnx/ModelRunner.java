package com.example.onnx;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.Arrays;

/**
 * {@link OnnxPredictor} をCLIから手軽に試すためのエントリーポイント。
 */
public final class ModelRunner {

    public static void main(String[] args) {
        // 引数が足りない場合は使い方を表示して終了
        if (args.length < 4) {
            printUsageAndExit();
        }

        String modelPath = args[0];
        String inputName = args[1];
        String outputName = args[2];
        // CSVモードの場合は --csv <csv-path> <columns>
        if ("--csv".equals(args[3])) {
            if (args.length < 6) {
                printUsageAndExit();
            }
            String csvPath = args[4];
            String[] columns = args[5].split(",");
            runCsvMode(modelPath, inputName, outputName, csvPath, columns);
            return;
        }

        // 4番目の引数はカンマ区切りの数値文字列なのでfloat配列に変換
        float[] values = parseFloats(args[3]);
        // shape引数が無ければバッチサイズ1＋特徴量数で推論する
        long[] shape = args.length >= 5 ? parseLongs(args[4]) : new long[]{1, values.length};

        // try-with-resourcesでPredictorを利用し、終わったらclose()を自動呼び出し
        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) {
            InferenceResult result = predictor.runInference(inputName, values, shape, outputName);
            // 推論結果と、計測した実行時間を表示
            System.out.println("Predictions: " + Arrays.toString(result.output()));
            System.out.printf("Inference time: %.3f ms (%d ns)%n", result.elapsedMillis(), result.elapsedNanos());
        } catch (IOException | OrtException ex) {
            System.err.println("Failed to run inference: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(2);
        }
    }

    private static float[] parseFloats(String csv) {
        String[] parts = csv.split(",");
        float[] values = new float[parts.length];
        // 文字列を1つずつfloatに変換して格納する
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    private static long[] parseLongs(String csv) {
        String[] parts = csv.split(",");
        long[] values = new long[parts.length];
        // shapeはlongで扱うためlongに変換
        for (int i = 0; i < parts.length; i++) {
            values[i] = Long.parseLong(parts[i].trim());
        }
        return values;
    }

    private static void runCsvMode(String modelPath,
                                   String inputName,
                                   String outputName,
                                   String csvPath,
                                   String[] columns) {
        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) {
            java.nio.file.Path path = java.nio.file.Path.of(csvPath);
            java.util.List<String> lines = java.nio.file.Files.readAllLines(path);
            if (lines.isEmpty()) {
                System.err.println("CSVが空です: " + csvPath);
                return;
            }

            // ヘッダー行から列名→インデックスのマップを作る
            String[] headers = lines.get(0).split(",");
            java.util.Map<String, Integer> indexByName = new java.util.HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                indexByName.put(headers[i].trim(), i);
            }

            for (int lineIdx = 1; lineIdx < lines.size(); lineIdx++) {
                String line = lines.get(lineIdx);
                if (line.isBlank()) {
                    continue;
                }
                String[] parts = line.split(",");
                float[] values = new float[columns.length];
                for (int c = 0; c < columns.length; c++) {
                    Integer idx = indexByName.get(columns[c].trim());
                    if (idx == null || idx >= parts.length) {
                        throw new IllegalArgumentException("列 " + columns[c] + " がCSVに見つかりません");
                    }
                    values[c] = Float.parseFloat(parts[idx].trim());
                }
                long[] shape = new long[]{1, values.length};
                InferenceResult result = predictor.runInference(inputName, values, shape, outputName);
                System.out.println("Row " + lineIdx + " -> " + java.util.Arrays.toString(result.output())
                        + " | time: " + String.format("%.3f ms", result.elapsedMillis()));
            }
        } catch (Exception ex) {
            System.err.println("Failed to run CSV inference: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(2);
        }
    }

    private static void printUsageAndExit() {
        System.err.println("""
                使い方:
                  1) 単一サンプル:
                     ModelRunner <model-path> <input-name> <output-name> <カンマ区切りの値> [shape]
                       例) ModelRunner model.onnx float_input probabilities 3,1,29,0,0,7.25,0 1,7

                  2) CSV一括推論:
                     ModelRunner <model-path> <input-name> <output-name> --csv <csv-path> <columns>
                       columns: CSVから読み出す列名をカンマ区切りで順序付きに指定 (例: Pclass,Sex,Age,SibSp,Parch,Fare,Embarked)
                       例) ModelRunner model.onnx float_input probabilities --csv data.csv Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
                """);
        System.exit(1);
    }
}
