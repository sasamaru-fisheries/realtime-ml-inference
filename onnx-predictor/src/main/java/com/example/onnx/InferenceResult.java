package com.example.onnx;

/**
 * 推論結果と所要時間(ナノ秒)をまとめて扱うためのDTO。
 * レコードを使うことでフィールド定義とgetterを自動生成している。
 */
public record InferenceResult(float[] output, long elapsedNanos) {
    /**
     * 経過時間をミリ秒に変換して返すユーティリティ。
     */
    public double elapsedMillis() {
        return elapsedNanos / 1_000_000.0;
    }
}
