package com.example.onnx;

import ai.onnxruntime.OrtException;
import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * 列名付きの入力をそのままONNXに渡すCSVバッチ推論用ランナー。
 * スキーマ(YAML)に numeric / categorical の列名を指定し、前処理はモデル側（ONNX内のOneHot/Imputer等）に任せる。
 *
 * 使い方:
 *   ModelRunner <model.onnx> <output-name> --csv <csv-path> <schema.yaml> [output-csv]
 *   例) ModelRunner model.onnx probabilities --csv data.csv schema.yaml preds.csv
 */
public final class ModelRunner {

    public static void main(String[] args) {
        if (args.length < 5 || !"--csv".equals(args[2])) {
            printUsageAndExit();
        }
        String modelPath = args[0];
        String outputName = args[1];
        String csvPath = args[3];
        String schemaPath = args[4];
        String outputCsv = args.length >= 6 ? args[5] : null;

        runCsvMode(modelPath, outputName, csvPath, schemaPath, outputCsv);
    }

    private static void runCsvMode(String modelPath,
                                   String outputName,
                                   String csvPath,
                                   String schemaPath,
                                   String outputCsv) {
        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) {
            Schema schema = Schema.load(Path.of(schemaPath));
            java.nio.file.Path path = java.nio.file.Path.of(csvPath);
            // ファイル読み込みや推論で例外が出た場合はまとめてcatchして終了
            List<String> lines = java.nio.file.Files.readAllLines(path);
            if (lines.isEmpty()) {
                System.err.println("CSVが空です: " + csvPath);
                return;
            }

            // ヘッダー行から列名→インデックスのマップを作る（小文字化して一致判定を緩める）
            String[] headers = lines.get(0).split(",");
            Map<String, Integer> indexByName = new HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                indexByName.put(headers[i].trim().toLowerCase(), i);
            }

            String[] columns = schema.columns();
            List<String> outputLines = new ArrayList<>();
            // ヘッダー: rowと各クラス確率に加え、time_ms/time_nsを出力
            outputLines.add("row" + buildHeader(schema.columns().length) + ",time_ms,time_ns");
            int row = 0;
            for (int lineIdx = 1; lineIdx < lines.size(); lineIdx++) {
                String line = lines.get(lineIdx);
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                Map<String, Object> inputMap = new HashMap<>();
                for (String col : columns) {
                    String keyLower = col.toLowerCase();
                    Integer idx = indexByName.get(keyLower);
                    if (idx == null || idx >= parts.length) continue;
                    String raw = parts[idx].trim();
                    // 数値はパース失敗時0にフォールバック、カテゴリは文字列のまま渡す
                    if (schema.isNumeric(col)) {
                        inputMap.put(col, new float[]{parseFloatSafe(raw)});
                    } else {
                        inputMap.put(col, new String[]{raw});
                    }
                }
                InferenceResult result = predictor.runInference(inputMap, outputName);
                float[] out = result.output();
                System.out.printf("Row %d -> %s | time: %.3f ms (%d ns)%n",
                        lineIdx, Arrays.toString(out), result.elapsedMillis(), result.elapsedNanos());
                outputLines.add(buildLine(lineIdx, out, result));
                row++;
            }
            if (outputCsv != null && outputLines.size() > 1) {
                java.nio.file.Path outPath = java.nio.file.Path.of(outputCsv);
                java.nio.file.Files.createDirectories(outPath.getParent() == null ? java.nio.file.Path.of(".") : outPath.getParent());
                java.nio.file.Files.write(outPath, outputLines);
                System.out.println("Saved predictions to " + outPath.toAbsolutePath());
            }
        } catch (Exception ex) {
            // どこで例外が出たかを表示した上で停止
            System.err.println("Failed to run CSV inference: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(2);
        }
    }

    private static float parseFloatSafe(String raw) {
        try {
            return Float.parseFloat(raw);
        } catch (NumberFormatException ex) {
            return 0f;
        }
    }

    private static String buildHeader(int classes) {
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < classes; c++) {
            sb.append(",prob_").append(c);
        }
        return sb.toString();
    }

    private static String buildLine(int row, float[] preds, InferenceResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append(row);
        for (float p : preds) {
            sb.append(",").append(p);
        }
        sb.append(",").append(String.format("%.3f", result.elapsedMillis()));
        sb.append(",").append(result.elapsedNanos());
        return sb.toString();
    }

    private static void printUsageAndExit() {
        System.err.println("""
                使い方:
                  CSV一括推論（前処理はモデルに任せる）:
                     ModelRunner <model.onnx> <output-name> --csv <csv-path> <schema.yaml> [output-csv]
                       schema.yaml: numeric/categorical の列名を持つYAML（train側と合わせる）
                       output-csv: 指定すると推論結果をCSVに保存
                       例) ModelRunner model.onnx probabilities --csv data.csv schema.yaml preds.csv
                """);
        System.exit(1);
    }

    private record Schema(List<String> numeric, List<String> categorical) {
        static Schema load(Path schemaPath) throws IOException {
            Yaml yaml = new Yaml();
            Map<String, Object> map = yaml.load(Files.readString(schemaPath));
            List<String> num = extract(map.get("numeric"));
            List<String> cat = extract(map.get("categorical"));
            return new Schema(num, cat);
        }

        private static List<String> extract(Object obj) {
            if (obj == null) return List.of();
            List<?> raw = (List<?>) obj;
            List<String> out = new ArrayList<>();
            for (Object o : raw) {
                if (o instanceof Map<?,?> m && m.containsKey("name")) {
                    out.add(String.valueOf(m.get("name")));
                } else {
                    out.add(String.valueOf(o));
                }
            }
            return out;
        }

        boolean isNumeric(String col) {
            return numeric.stream().anyMatch(c -> c.equalsIgnoreCase(col));
        }

        String[] columns() {
            List<String> all = new ArrayList<>();
            all.addAll(numeric);
            all.addAll(categorical);
            return all.toArray(new String[0]);
        }
    }
}
