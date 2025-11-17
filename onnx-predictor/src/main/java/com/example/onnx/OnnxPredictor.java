package com.example.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Optional;
import java.util.HashMap;

/**
 * インスタンス化時にONNXモデルを読み込み、推論とリロード機能を提供する軽量ラッパー。
 * Pythonの「学習済みモデルクラス」と同じイメージで利用できる。
 */
public class OnnxPredictor implements Closeable {
    private final OrtEnvironment environment;
    private final Path modelPath;
    private OrtSession session;

    public OnnxPredictor(String modelPath) throws IOException, OrtException {
        // Pathオブジェクトに変換し、入力されたパスが正しいかを確認する
        this.modelPath = Paths.get(modelPath);
        // モデルファイルの存在を先に確認しておく
        if (!Files.exists(this.modelPath)) {
            throw new IOException("Model not found at: " + this.modelPath);
        }
        // ONNX Runtimeのグローバル環境を取得（GPU/CPU管理などを担当）
        this.environment = OrtEnvironment.getEnvironment();
        // セッションを生成して推論準備を整える
        loadSession();
    }

    private synchronized void loadSession() throws OrtException {
        // 既にセッションが生成済みなら一旦閉じてから作り直す
        closeSession();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        this.session = environment.createSession(modelPath.toString(), options);
    }

    private void closeSession() {
        if (this.session != null) {
            try {
                this.session.close();
            } catch (OrtException ex) {
                throw new IllegalStateException("Failed to close ONNX session", ex);
            } finally {
                this.session = null;
            }
        }
    }

    public synchronized void reloadModel() throws OrtException {
        loadSession();
    }

    /**
     * ONNX Runtimeに入力テンソルを与えて推論を実行し、所要時間も返す。
     */
    public synchronized InferenceResult runInference(String inputName,
                                                     float[] flattenedInput,
                                                     long[] inputShape,
                                                     String outputName) throws OrtException {
        // セッションがnullのままなら利用者側の不具合なので例外を投げる
        if (session == null) {
            throw new IllegalStateException("ONNX session is not initialized");
        }
        // 推論開始時刻を取得（ナノ秒単位）
        long start = System.nanoTime();
        // try-with-resourcesでテンソルと推論結果を自動クローズ
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(flattenedInput), inputShape);
             OrtSession.Result result = session.run(Map.of(inputName, inputTensor))) {
            // 出力名に一致する結果をOptionalで受け取り、存在しない場合は例外にする
            Optional<OnnxValue> optionalValue = result.get(outputName);
            OnnxValue outputValue = optionalValue.orElseThrow(
                () -> new IllegalStateException("Output '" + outputName + "' not found in inference result")
            );
            // 期待したテンソル型かどうかをチェック
            if (!(outputValue instanceof OnnxTensor tensor)) {
                throw new IllegalStateException("Expected tensor output but received: " + outputValue.getClass());
            }
            // 結果をfloat配列に変換し、計測時間とセットで返す
            float[] output = toFloatArray(tensor);
            long duration = System.nanoTime() - start;
            return new InferenceResult(output, duration);
        }
    }

    /**
     * 複数入力（列ごとのTensorなど）を受け取って推論を実行する。
     * 値は float[] または String[] を想定しており、内部でOnnxTensorに変換する。
     */
    public synchronized InferenceResult runInference(Map<String, Object> inputs,
                                                     String outputName) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("ONNX session is not initialized");
        }
        Map<String, OnnxTensor> tensorMap = new HashMap<>();
        long start = System.nanoTime();
        try {
            for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                tensorMap.put(entry.getKey(), createTensor(entry.getValue()));
            }
            try (OrtSession.Result result = session.run(tensorMap)) {
                OnnxValue outputValue = result.get(outputName).orElseThrow(
                        () -> new IllegalStateException("Output '" + outputName + "' not found in inference result")
                );
                if (!(outputValue instanceof OnnxTensor tensor)) {
                    throw new IllegalStateException("Expected tensor output but received: " + outputValue.getClass());
                }
                float[] output = toFloatArray(tensor);
                long duration = System.nanoTime() - start;
                return new InferenceResult(output, duration);
            }
        } finally {
            for (OnnxTensor tensor : tensorMap.values()) {
                if (tensor != null) {
                    tensor.close();
                }
            }
        }
    }

    private OnnxTensor createTensor(Object value) throws OrtException {
        if (value instanceof float[] floats) {
            return OnnxTensor.createTensor(environment, FloatBuffer.wrap(floats), new long[]{floats.length, 1});
        }
        if (value instanceof String[] strings) {
            return OnnxTensor.createTensor(environment, strings, new long[]{strings.length, 1});
        }
        throw new IllegalArgumentException("Unsupported input type: " + value.getClass());
    }

    private float[] toFloatArray(OnnxTensor tensor) throws OrtException {
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[] copy = new float[buffer.remaining()];
        buffer.get(copy);
        return copy;
    }

    @Override
    public synchronized void close() {
        // セッションと環境を順番に閉じる
        closeSession();
        environment.close();
    }
}
