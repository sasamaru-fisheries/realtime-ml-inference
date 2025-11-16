package com.example.pmml;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * PmmlPredictor をライブラリとして使うときの簡易サンプル。
 *
 * 使い方:
 *   java -cp target/pmml-predictor-1.0.0.jar com.example.pmml.SampleUsage <model1.pmml> [model2.pmml]
 *
 * model2 を省略すると同じモデルを再利用します。
 */
public final class SampleUsage {
    private static final String[] FEATURE_ORDER = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"};

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("使い方: SampleUsage <model1.pmml> [model2.pmml]");
            System.exit(1);
        }
        String model1 = args[0];
        String model2 = args.length >= 2 ? args[1] : null;

        try (PmmlPredictor predictor = new PmmlPredictor(model1)) {
            Map<String, Object> input1 = buildInput(3f, "male", 29f, 0f, 0f, 7.25f, "S");
            InferenceResult r1 = predictor.runInference(input1);
            System.out.println("Model1 label: " + r1.predictedLabel());
            System.out.println("Model1 prob : " + r1.probabilities());
            System.out.printf("Elapsed: %.3f ms%n", r1.elapsedMillis());

            if (model2 != null) {
                try (PmmlPredictor predictor2 = new PmmlPredictor(model2)) {
                    Map<String, Object> input2 = buildInput(1f, "female", 35f, 1f, 0f, 53.1f, "C");
                    InferenceResult r2 = predictor2.runInference(input2);
                    System.out.println("Model2 label: " + r2.predictedLabel());
                    System.out.println("Model2 prob : " + r2.probabilities());
                    System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis());
                }
            } else {
                Map<String, Object> input2 = buildInput(1f, "female", 35f, 1f, 0f, 53.1f, "C");
                InferenceResult r2 = predictor.runInference(input2);
                System.out.println("Reloaded label: " + r2.predictedLabel());
                System.out.println("Reloaded prob : " + r2.probabilities());
                System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis());
            }
        }
    }

    private static Map<String, Object> buildInput(Object... values) {
        if (values.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException("Expected " + FEATURE_ORDER.length + " values");
        }
        Map<String, Object> map = new HashMap<>();
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            map.put(FEATURE_ORDER[i], values[i]);
        }
        return map;
    }
}
