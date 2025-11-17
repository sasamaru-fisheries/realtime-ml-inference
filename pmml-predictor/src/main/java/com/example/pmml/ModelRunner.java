package com.example.pmml;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * スキーマ(YAML)を読み込み、前処理はモデル側に任せて推論するPMMLランナー。
 *
 * 使い方:
 *   ModelRunner <pmml-path> --csv <csv-path> <schema.yaml> [output-csv]
 *   例) ModelRunner model.pmml --csv data.csv schema.yaml preds.csv
 */
public final class ModelRunner {

    public static void main(String[] args) {
        if (args.length < 4 || !"--csv".equals(args[1])) {
            printUsageAndExit();
        }
        String pmmlPath = args[0];
        String csvPath = args[2];
        String schemaPath = args[3];
        String outputCsv = args.length >= 5 ? args[4] : null;

        runCsv(pmmlPath, csvPath, schemaPath, outputCsv);
    }

    private static void runCsv(String pmmlPath, String csvPath, String schemaPath, String outputCsv) {
        try (PmmlPredictor predictor = new PmmlPredictor(pmmlPath)) {
            Schema schema = Schema.load(Path.of(schemaPath));
            Path path = Path.of(csvPath);
            Reader reader = Files.newBufferedReader(path);
            CSVParser parser = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
            Map<String, Integer> indexByName = new HashMap<>();
            parser.getHeaderMap().forEach((k, v) -> indexByName.put(k.trim().toLowerCase(), v));

            List<String> outLines = new ArrayList<>();
            outLines.add("row,prob_0,prob_1,time_ms,time_ns");

            int row = 0;
            int lineIdx = 1;
            for (CSVRecord record : parser) {
                Map<String, Object> inputMap = new HashMap<>();
                for (String col : schema.columns()) {
                    String key = col.toLowerCase();
                    Integer idx = indexByName.get(key);
                    if (idx == null) {
                        throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません");
                    }
                    String raw = record.get(idx).trim();
                    if (schema.isNumeric(col)) {
                        inputMap.put(col, parseFloatSafe(raw));
                    } else {
                        inputMap.put(col, raw);
                    }
                }
                InferenceResult result = predictor.runInference(inputMap);
                Map<String, Double> probs = result.probabilities();
                double prob0 = probs.getOrDefault("0", 0.0);
                double prob1 = probs.getOrDefault("1", 0.0);
                outLines.add((lineIdx) + "," + prob0 + "," + prob1
                        + "," + String.format("%.3f", result.elapsedMillis()) + "," + result.elapsedNanos());
                System.out.printf("Row %d -> prob0=%.4f, prob1=%.4f, time=%.3f ms%n",
                        lineIdx, prob0, prob1, result.elapsedMillis());
                row++;
                lineIdx++;
            }
            if (outputCsv != null) {
                Path outPath = Path.of(outputCsv);
                Files.createDirectories(outPath.getParent() == null ? Path.of(".") : outPath.getParent());
                Files.write(outPath, outLines);
                System.out.println("Saved PMML predictions to " + outPath.toAbsolutePath());
            }
            System.out.println("Processed rows: " + row);
        } catch (Exception ex) {
            System.err.println("Failed to run PMML CSV inference: " + ex.getMessage());
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

    private static void printUsageAndExit() {
        System.err.println("""
                使い方:
                  CSV一括推論（前処理はモデル側）:
                     ModelRunner <pmml-path> --csv <csv-path> <schema.yaml> [output-csv]
                       schema.yaml: numeric/categorical の列名を持つYAML（train側と合わせる）
                       output-csv: 指定すると予測結果をCSVに保存
                       例) ModelRunner model.pmml --csv data.csv schema.yaml preds.csv
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
