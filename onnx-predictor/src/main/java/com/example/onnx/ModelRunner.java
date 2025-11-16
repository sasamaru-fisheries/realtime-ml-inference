package com.example.onnx;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.Arrays;

/**
 * {@link OnnxPredictor} をCLIから手軽に試すためのエントリーポイント。
 */
public final class ModelRunner {
    private static final java.util.Set<String> NUMERIC_COLUMNS_LOWER =
            java.util.Set.of("pclass", "age", "sibsp", "parch", "fare");
    private static final java.util.Set<String> CATEGORICAL_COLUMNS_LOWER =
            java.util.Set.of("sex", "embarked");
    private static final java.util.Map<String, Float> SEX_MAP =
            java.util.Map.of("male", 1.0f, "female", 0.0f);
    private static final java.util.Map<String, Float> EMBARKED_MAP =
            java.util.Map.of("s", 0.0f, "c", 1.0f, "q", 2.0f);

    public static void main(String[] args) {
        // 引数が足りない場合は使い方を表示して終了
        if (args.length < 4) {
            printUsageAndExit();
        }

        String modelPath = args[0];
        String inputName = args[1];
        String outputName = args[2];
        // CSVモードの場合は --csv <csv-path> <columns> [output-csv]
        if ("--csv".equals(args[3])) {
            if (args.length < 6) {
                printUsageAndExit();
            }
            String csvPath = args[4];
            String[] columns = args[5].split(",");
            String outputCsv = args.length >= 7 ? args[6] : null;
            runCsvMode(modelPath, inputName, outputName, csvPath, columns, outputCsv);
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
                                   String[] columns,
                                   String outputCsv) {
        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) {
            java.nio.file.Path path = java.nio.file.Path.of(csvPath);
            java.util.List<String> lines = java.nio.file.Files.readAllLines(path);
            if (lines.isEmpty()) {
                System.err.println("CSVが空です: " + csvPath);
                return;
            }

            // ヘッダー行から列名→インデックスのマップを作る（小文字化して一致判定を緩める）
            String[] headers = lines.get(0).split(",");
            java.util.Map<String, Integer> indexByName = new java.util.HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                indexByName.put(headers[i].trim().toLowerCase(), i);
            }

            int batch = lines.size() - 1;
            int row = 0;
            java.util.List<String> outputLines = new java.util.ArrayList<>();
            int classes = -1;
            for (int lineIdx = 1; lineIdx < lines.size(); lineIdx++) {
                String line = lines.get(lineIdx);
                if (line.isBlank()) {
                    continue;
                }
                String[] parts = line.split(",");
                float[] values = new float[columns.length];
                for (int c = 0; c < columns.length; c++) {
                    String trimmed = columns[c].trim();
                    String key = trimmed.toLowerCase();
                    Integer idx = indexByName.get(key);
                    if (idx == null || idx >= parts.length) {
                        throw new IllegalArgumentException("列 " + trimmed + " がCSVに見つかりません");
                    }
                    String raw = parts[idx].trim();
                    if (CATEGORICAL_COLUMNS_LOWER.contains(key)) {
                        values[c] = mapCategory(key, raw);
                    } else if (NUMERIC_COLUMNS_LOWER.contains(key)) {
                        values[c] = parseFloatSafe(raw);
                    } else {
                        throw new IllegalArgumentException("列 " + trimmed + " の型が判定できません（数値かカテゴリか指定してください）");
                    }
                }
                InferenceResult result = predictor.runInference(inputName, values, new long[]{1, values.length}, outputName);
                System.out.println("Row " + (lineIdx) + " -> " + java.util.Arrays.toString(result.output())
                        + " | time: " + String.format("%.3f ms", result.elapsedMillis()));
                if (classes == -1) {
                    classes = result.output().length;
                    StringBuilder header = new StringBuilder("row");
                    for (int c = 0; c < classes; c++) {
                        header.append(",prob_").append(c);
                    }
                    outputLines.add(header.toString());
                }
                StringBuilder lineOut = new StringBuilder();
                lineOut.append(lineIdx);
                float[] preds = result.output();
                for (float p : preds) {
                    lineOut.append(",").append(p);
                }
                outputLines.add(lineOut.toString());
                row++;
            }
            if (outputCsv != null && !outputLines.isEmpty()) {
                java.nio.file.Path outPath = java.nio.file.Path.of(outputCsv);
                java.nio.file.Files.createDirectories(outPath.getParent() == null ? java.nio.file.Path.of(".") : outPath.getParent());
                java.nio.file.Files.write(outPath, outputLines);
                System.out.println("Saved predictions to " + outPath.toAbsolutePath());
            }
        } catch (Exception ex) {
            System.err.println("Failed to run CSV inference: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(2);
        }
    }

    private static float mapCategory(String key, String raw) {
        String lower = raw.toLowerCase();
        if ("sex".equals(key)) {
            return SEX_MAP.getOrDefault(lower, 0f);
        }
        if ("embarked".equals(key)) {
            return EMBARKED_MAP.getOrDefault(lower, 0f);
        }
        return 0f;
    }

    private static float parseFloatSafe(String raw) {
        try {
            return Float.parseFloat(raw);
        } catch (NumberFormatException ex) {
            return 0f;
        }
    }

    private static void printUsageAndExit() {
        System.err.println("""
                使い方:
                  1) 単一サンプル:
                     ModelRunner <model-path> <input-name> <output-name> <カンマ区切りの値> [shape]
                       例) ModelRunner model.onnx float_input probabilities 3,1,29,0,0,7.25,0 1,7
                       （※ 前処理込みONNXでは列ごとの入力を推奨、CSVモード推奨）

                  2) CSV一括推論:
                     ModelRunner <model-path> <input-name> <output-name> --csv <csv-path> <columns> [output-csv]
                       columns: CSVから読み出す列名をカンマ区切りで順序付きに指定 (例: Pclass,Sex,Age,SibSp,Parch,Fare,Embarked)
                       output-csv: 指定すると推論結果をCSVに保存
                       例) ModelRunner model.onnx float_input probabilities --csv data.csv Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
                """);
        System.exit(1);
    }
}
