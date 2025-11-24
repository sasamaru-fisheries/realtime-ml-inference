package com.example.pmml; // pmmlパッケージの推論結果DTO

import java.util.Map; // クラスラベルと確率を保持するためMapを利用

/**
 * PMML推論の結果を保持するDTO。ラベルと確率分布、実行時間をセットで返す。 // DTOの役割を説明
 */
public record InferenceResult(String predictedLabel, Map<String, Double> probabilities, long elapsedNanos) { // 予測ラベル・確率分布・経過ナノ秒を保持するレコード
    public double elapsedMillis() { // 経過時間をミリ秒に変換して返すメソッド
        return elapsedNanos / 1_000_000.0; // ナノ秒をミリ秒に変換
    }
} // InferenceResultレコードの終端
