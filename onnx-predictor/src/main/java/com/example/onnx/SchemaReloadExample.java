package com.example.onnx;

import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * スキーマ(YAML)を読み込み、CSVの任意行（またはデフォルト値）で
 * 複数モデルを同一JVM内で順に推論するサンプル。
 *
 * 使い方:
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SchemaReloadExample \
 *     <model1.onnx> <model2.onnx> <schema.yaml> [csv] [rowIndex(1-based)] [output-name]
 *
 * 例:
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SchemaReloadExample \
 *     ../models/model1.onnx ../models/model2.onnx ../schema.yaml ../data/input.csv 2 probabilities
 */
public final class SchemaReloadExample {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("""
                    使い方:
                      SchemaReloadExample <model1.onnx> <model2.onnx> <schema.yaml> [csv] [rowIndex] [output-name]
                        csv: (任意) 入力データのCSV。指定しない場合はスキーマに従ってデフォルト(0/空文字)を使用
                        rowIndex: (任意) CSVの何行目を使うか（1始まり、ヘッダーを含めない）
                        output-name: (任意) ONNX出力名。省略時は probabilities
                    """);
            System.exit(1);
        }
        // モデル2つ・スキーマ・任意のCSV/出力名を取得
        String model1 = args[0];
        String model2 = args[1];
        String schemaPath = args[2];
        String csvPath = args.length >= 4 ? args[3] : null;
        int rowIndex = args.length >= 5 ? Integer.parseInt(args[4]) : 1;
        String outputName = args.length >= 6 ? args[5] : "probabilities";

        // スキーマをロードし、CSV指定があれば該当行、なければデフォルト値で入力マップを構築
        Schema schema = Schema.load(Path.of(schemaPath));
        Map<String, Object> input = csvPath != null
                ? loadRowFromCsv(csvPath, schema, rowIndex)
                : buildDefaultInputs(schema);

        // 1つのPredictorインスタンスを再利用し、モデル1→モデル2へ差し替えて推論を実行
        try (OnnxPredictor predictor = new OnnxPredictor(model1)) {
            InferenceResult r1 = predictor.runInference(input, outputName);
            System.out.println("Model1 output: " + Arrays.toString(r1.output()));
            System.out.printf("Elapsed: %.3f ms%n", r1.elapsedMillis());

            predictor.setModelPath(model2); // model2に差し替え
            predictor.reloadModel();        // セッションを再構築

            InferenceResult r2 = predictor.runInference(input, outputName);
            System.out.println("Model2 output: " + Arrays.toString(r2.output()));
            System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis());
        }
    }

    private static Map<String, Object> buildDefaultInputs(Schema schema) {
        // スキーマの列に合わせて0/空文字を仕込む
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
        // CSVを全行読み込んで対象行を探す（簡易実装）
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
