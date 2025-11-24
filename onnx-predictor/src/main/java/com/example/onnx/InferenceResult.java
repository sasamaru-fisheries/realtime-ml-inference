package com.example.onnx; // onnxパッケージに属するDTOであることを示す

/**
 * 推論結果と所要時間(ナノ秒)をまとめて扱うためのDTO。 // 出力と経過時間をまとめる目的を説明
 * レコードを使うことでフィールド定義とgetterを自動生成している。 // record構文でボイラープレートを省略していることを示す
 */
public record InferenceResult(float[] output, long elapsedNanos) { // 推論出力配列と経過ナノ秒を保持するレコード
    /**
     * 経過時間をミリ秒に変換して返すユーティリティ。 // ナノ秒を扱いやすいミリ秒に変換する補助メソッド
     */
    public double elapsedMillis() { // ミリ秒単位の経過時間を返すメソッド
        return elapsedNanos / 1_000_000.0; // ナノ秒をミリ秒に変換して返却
    }
} // InferenceResultレコードの終端
