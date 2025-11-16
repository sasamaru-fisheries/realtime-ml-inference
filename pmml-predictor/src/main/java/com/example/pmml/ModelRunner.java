package com.example.pmml;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PMMLモデルをCLIから手軽に試すためのエントリーポイント。
 */
public final class ModelRunner {

    // Titanicモデルで利用する特徴量名の順序。PMML側と一致させる必要がある。
    private static final List<String> FEATURE_ORDER = List.of(
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
    );

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("""
                    使い方: ModelRunner <pmml-path> <comma-separated-values> <target-name>
                      pmml-path: PMMLファイルへのパス
                      comma-separated-values: 特徴量をカンマ区切りで指定 (例: 3,1,29,0,0,7.25,0)
                      target-name: 予測対象のフィールド名 (例: Survived)
                    """);
            System.exit(1);
        }

        String pmmlPath = args[0];
        float[] values = parseFloats(args[1]);
        String targetName = args[2];

        if (values.length != FEATURE_ORDER.size()) {
            System.err.println("特徴量の数がPMMLモデルと一致していません。期待する数: " + FEATURE_ORDER.size());
            System.exit(1);
        }

        Map<String, Object> inputs = buildInputMap(values);

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
}
