package com.example.onnx;

import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * スキーマYAMLを読み込み、CSVの任意行（またはデフォルト値）を使って推論するサンプル。
 *
 * 使い方:
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SampleUsage \
 *     <model.onnx> <output-name|省略でprobabilities> <schema.yaml> [csv-path] [rowIndex(1-based)]
 *
 * 例:
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SampleUsage \
 *     ../models/titanic_random_forest.onnx probabilities ../schema.yaml ../data/Titanic-Dataset.csv 2
 */
public final class SampleUsage {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("""
                    使い方:
                      SampleUsage <model.onnx> <output-name|省略可> <schema.yaml> [csv] [rowIndex]
                        output-name: 省略時は probabilities
                        csv: (任意) 入力データのCSV。指定しない場合はスキーマで0/空文字のダミーを作成
                        rowIndex: (任意) CSVの何行目を使うか（1始まり、ヘッダーを含めない）
                    """);
            System.exit(1);
        }

        String modelPath = args[0];
        String outputName;
        String schemaPath;
        String csvPath = null;
        int rowIndex = 1;

        if (args.length >= 3) {
            outputName = args[1];
            schemaPath = args[2];
            if (args.length >= 4) {
                csvPath = args[3];
            }
            if (args.length >= 5) {
                rowIndex = Integer.parseInt(args[4]);
            }
        } else {
            // output-name省略パターン: <model> <schema> ...
            outputName = "probabilities";
            schemaPath = args[1];
            if (args.length >= 3) {
                csvPath = args[2];
            }
            if (args.length >= 4) {
                rowIndex = Integer.parseInt(args[3]);
            }
        }

        Schema schema = Schema.load(Path.of(schemaPath));
        Map<String, Object> inputMap = csvPath != null
                ? loadRowFromCsv(csvPath, schema, rowIndex)
                : buildDefaultInputs(schema);

        try (OnnxPredictor predictor = new OnnxPredictor(modelPath)) {
            InferenceResult r = predictor.runInference(inputMap, outputName);
            System.out.println("Output (" + outputName + "): " + Arrays.toString(r.output()));
            System.out.printf("Elapsed: %.3f ms (%d ns)%n", r.elapsedMillis(), r.elapsedNanos());
        }
    }

    private static Map<String, Object> buildDefaultInputs(Schema schema) {
        Map<String, Object> map = new HashMap<>();
        for (String col : schema.numeric()) {
            map.put(col, new float[]{0f});
        }
        for (String col : schema.categorical()) {
            map.put(col, new String[]{""});
        }
        return map;
    }

    private static Map<String, Object> loadRowFromCsv(String csvPath, Schema schema, int rowIndex) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(csvPath));
        if (lines.size() < 2) {
            throw new IllegalArgumentException("CSVが空かヘッダーのみです: " + csvPath);
        }
        if (rowIndex < 1 || rowIndex >= lines.size()) {
            throw new IllegalArgumentException("rowIndex が範囲外です (1.." + (lines.size() - 1) + ")");
        }
        String[] headers = lines.get(0).split(",");
        Map<String, Integer> indexByName = new HashMap<>();
        for (int i = 0; i < headers.length; i++) {
            indexByName.put(headers[i].trim().toLowerCase(), i);
        }
        String[] parts = lines.get(rowIndex).split(",");

        Map<String, Object> inputMap = new HashMap<>();
        for (String col : schema.columns()) {
            String keyLower = col.toLowerCase();
            Integer idx = indexByName.get(keyLower);
            if (idx == null || idx >= parts.length) {
                throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません");
            }
            String raw = parts[idx].trim();
            if (schema.isNumeric(col)) {
                inputMap.put(col, new float[]{parseFloatSafe(raw)});
            } else {
                inputMap.put(col, new String[]{raw});
            }
        }
        return inputMap;
    }

    private static float parseFloatSafe(String raw) {
        try {
            return Float.parseFloat(raw);
        } catch (NumberFormatException ex) {
            return 0f;
        }
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
                if (o instanceof Map<?, ?> m && m.containsKey("name")) {
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
