package com.example.onnx;

import java.util.Arrays;

/**
 * OnnxPredictor をライブラリとして使うときの簡易サンプル。
 *
 * 使い方:
 *   java -cp target/onnx-predictor-1.0.0.jar com.example.onnx.SampleUsage <model1.onnx> [model2.onnx]
 *
 * model2 を省略すると同じモデルを reload して再推論します。
 */
public final class SampleUsage {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("使い方: SampleUsage <model1.onnx> [model2.onnx]");
            System.exit(1);
        }
        String model1 = args[0];
        String model2 = args.length >= 2 ? args[1] : null;

        // ❶ コンストラクタでモデルロード
        try (OnnxPredictor predictor = new OnnxPredictor(model1)) {
            float[] input1 = {3f, 1f, 29f, 0f, 0f, 7.25f, 0f};
            long[] shape = {1, input1.length};

            // ❷ 推論
            InferenceResult r1 = predictor.runInference("float_input", input1, shape, "probabilities");
            System.out.println("Model1 predictions: " + Arrays.toString(r1.output()));
            System.out.printf("Elapsed: %.3f ms%n", r1.elapsedMillis());

            // ❸ モデルをリロード（同じパスなら熱リロード、別パスなら差し替え）
            if (model2 != null) {
                try (OnnxPredictor predictor2 = new OnnxPredictor(model2)) {
                    float[] input2 = {1f, 0f, 35f, 1f, 0f, 53.1f, 1f};
                    InferenceResult r2 = predictor2.runInference("float_input", input2, shape, "probabilities");
                    System.out.println("Model2 predictions: " + Arrays.toString(r2.output()));
                    System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis());
                }
            } else {
                predictor.reloadModel();
                float[] input2 = {1f, 0f, 35f, 1f, 0f, 53.1f, 1f};
                InferenceResult r2 = predictor.runInference("float_input", input2, shape, "probabilities");
                System.out.println("Reloaded predictions: " + Arrays.toString(r2.output()));
                System.out.printf("Elapsed: %.3f ms%n", r2.elapsedMillis());
            }
        }
    }
}
