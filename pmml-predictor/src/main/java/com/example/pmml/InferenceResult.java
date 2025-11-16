package com.example.pmml;

import java.util.Map;

/**
 * PMML推論の結果を保持するDTO。ラベルと確率分布、実行時間をセットで返す。
 */
public record InferenceResult(String predictedLabel, Map<String, Double> probabilities, long elapsedNanos) {
    public double elapsedMillis() {
        return elapsedNanos / 1_000_000.0;
    }
}
