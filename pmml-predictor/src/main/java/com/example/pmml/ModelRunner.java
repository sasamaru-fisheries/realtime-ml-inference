package com.example.pmml;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/**
 * PMMLモデルをCLIから手軽に試すためのエントリーポイント。
 */
public final class ModelRunner {

    // Titanicモデルで利用する特徴量名の順序。PMML側と一致させる必要がある。
    private static final List<String> FEATURE_ORDER = List.of(
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
    );
    private static final Set<String> CATEGORICAL = Set.of("sex", "embarked");
    private static final Set<String> NUMERIC = Set.of("pclass", "age", "sibsp", "parch", "fare");
    private static final Map<String, Float> SEX_MAP = Map.of("male", 1.0f, "female", 0.0f);
    private static final Map<String, Float> EMBARKED_MAP = Map.of("s", 0.0f, "c", 1.0f, "q", 2.0f);

    public static void main(String[] args) {
        if (args.length < 3) {
            printUsageAndExit();
        }

        String pmmlPath = args[0];
        // CSVモード
        if ("--csv".equals(args[1])) {
            if (args.length < 4) {
                printUsageAndExit();
            }
            String csvPath = args[2];
            String[] columns = args[3].split(",");
            String outputCsv = args.length >= 5 ? args[4] : null;
            runCsv(pmmlPath, csvPath, columns, outputCsv);
            return;
        }

        // 単一サンプル（カンマ区切り値）
        float[] values = parseFloats(args[1]);
        if (values.length != FEATURE_ORDER.size()) {
            System.err.println("特徴量の数がPMMLモデルと一致していません。期待する数: " + FEATURE_ORDER.size());
            System.exit(1);
        }
        Map<String, Object> inputs = buildInputMap(values);
        runSingle(pmmlPath, inputs, "Survived");
    }

    private static float[] parseFloats(String csv) {
        String[] parts = csv.split(",");
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }

    private static Map<String, Object> buildInputMap(float[] values) {
        Map<String, Object> map = new HashMap<>();
        for (int i = 0; i < FEATURE_ORDER.size(); i++) {
            map.put(FEATURE_ORDER.get(i), values[i]);
        }
        return map;
    }

    private static void runSingle(String pmmlPath, Map<String, Object> inputs, String targetName) {
        try (PmmlPredictor predictor = new PmmlPredictor(pmmlPath)) {
            InferenceResult result = predictor.runInference(inputs);
            System.out.println("Predicted label (" + targetName + "): " + result.predictedLabel());
            System.out.println("Probabilities: " + result.probabilities());
            System.out.printf("Inference time: %.3f ms (%d ns)%n", result.elapsedMillis(), result.elapsedNanos());
        } catch (IOException ex) {
            System.err.println("Failed to run PMML inference: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(2);
        }
    }

    private static void runCsv(String pmmlPath, String csvPath, String[] columns, String outputCsv) {
        try (PmmlPredictor predictor = new PmmlPredictor(pmmlPath)) {
            Path path = Path.of(csvPath);
            Reader reader = Files.newBufferedReader(path);
            CSVParser parser = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
            Map<String, Integer> indexByName = new HashMap<>();
            parser.getHeaderMap().forEach((k, v) -> indexByName.put(k.trim().toLowerCase(), v));
            List<String> outLines = new ArrayList<>();
            outLines.add("row,label,prob_0,prob_1");
            int row = 0;
            int lineIdx = 1;
            for (CSVRecord record : parser) {
                Map<String, Object> inputMap = new HashMap<>();
                for (int c = 0; c < columns.length; c++) {
                    String col = columns[c].trim();
                    String key = col.toLowerCase();
                    Integer idx = indexByName.get(key);
                    if (idx == null) {
                        throw new IllegalArgumentException("列 " + col + " がCSVに見つかりません");
                    }
                    String raw = record.get(idx).trim();
                    if (CATEGORICAL.contains(key)) {
                        inputMap.put(col, mapCategory(key, raw));
                    } else if (NUMERIC.contains(key)) {
                        inputMap.put(col, parseFloatSafe(raw));
                    } else {
                        throw new IllegalArgumentException("列 " + col + " の型が判定できません");
                    }
                }
                InferenceResult result = predictor.runInference(inputMap);
                Map<String, Double> probs = result.probabilities();
                double prob0 = probs.getOrDefault("0", 0.0);
                double prob1 = probs.getOrDefault("1", 0.0);
                outLines.add((lineIdx) + "," + result.predictedLabel() + "," + prob0 + "," + prob1);
                System.out.printf("Row %d -> label=%s, prob0=%.4f, prob1=%.4f, time=%.3f ms%n",
                        lineIdx, result.predictedLabel(), prob0, prob1, result.elapsedMillis());
                row++;
                lineIdx++;
            }
            if (outputCsv != null) {
                java.nio.file.Path outPath = java.nio.file.Path.of(outputCsv);
                java.nio.file.Files.createDirectories(outPath.getParent() == null ? java.nio.file.Path.of(".") : outPath.getParent());
                java.nio.file.Files.write(outPath, outLines);
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

    private static void printUsageAndExit() {
        System.err.println("""
                使い方:
                  1) 単一サンプル:
                     ModelRunner <pmml-path> <comma-separated-values> Survived
                       例) ModelRunner model.pmml 3,1,29,0,0,7.25,0 Survived

                  2) CSV一括推論:
                     ModelRunner <pmml-path> --csv <csv-path> <columns> [output-csv]
                       columns: CSVから読み出す列名（順序付き）例: Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
                       output-csv: 指定すると予測結果をCSVに保存
                       例) ModelRunner model.pmml --csv data.csv Pclass,Sex,Age,SibSp,Parch,Fare,Embarked preds.csv
                """);
        System.exit(1);
    }
}
