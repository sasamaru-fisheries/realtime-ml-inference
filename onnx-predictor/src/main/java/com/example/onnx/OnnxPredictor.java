package com.example.onnx; // onnxパッケージ配下に配置された推論ラッパークラス

import ai.onnxruntime.OnnxTensor; // ONNX RuntimeのTensor型を扱うためのインポート
import ai.onnxruntime.OnnxValue; // 汎用的なONNX出力値を扱うクラスをインポート
import ai.onnxruntime.OrtEnvironment; // ONNX Runtimeの環境設定を表すクラスをインポート
import ai.onnxruntime.OrtException; // ONNX Runtimeが投げる例外をインポート
import ai.onnxruntime.OrtSession; // 推論セッションを表すクラスをインポート

import java.io.Closeable; // クローズ可能なリソースを表すインターフェースをインポート
import java.io.IOException; // 入出力例外を扱うためのインポート
import java.nio.FloatBuffer; // floatのバッファを扱うためのインポート
import java.nio.file.Files; // ファイル存在確認などのユーティリティをインポート
import java.nio.file.Path; // ファイルパスを表すPathクラスをインポート
import java.nio.file.Paths; // 文字列からPathを生成するユーティリティをインポート
import java.util.Map; // Mapインターフェースをインポート
import java.util.Optional; // Optionalクラスをインポート
import java.util.HashMap; // 可変Map実装であるHashMapをインポート

/**
 * インスタンス化時にONNXモデルを読み込み、推論とリロード機能を提供する軽量ラッパー。 // 役割の概要を示すコメント
 * Pythonの「学習済みモデルクラス」と同じイメージで利用できる。 // Pythonのモデルクラスのように使えることを補足
 */
public class OnnxPredictor implements Closeable { // 推論処理とクローズを提供するクラス
    private final OrtEnvironment environment; // ONNX Runtimeの環境を保持
    private Path modelPath; // モデルファイルのパスを保持
    private OrtSession session; // 推論セッションを保持

    public OnnxPredictor(String modelPath) throws IOException, OrtException { // コンストラクタでモデルパスを受け取る
        this.environment = OrtEnvironment.getEnvironment(); // グローバルなONNX環境を取得
        setModelPathInternal(Paths.get(modelPath)); // モデルパスを内部設定し存在確認
        loadSession(); // セッションを作成して推論準備を行う
    }

    /**
     * 参照するONNXモデルファイルを変更する。reloadModel() を呼び出すことで新しいモデルをロードできる。 // モデル差し替えの説明
     */
    public synchronized void setModelPath(String newModelPath) throws IOException { // モデルパスを変更する公開メソッド
        setModelPathInternal(Paths.get(newModelPath)); // 文字列パスをPath化して内部設定する
    }

    private void setModelPathInternal(Path newPath) throws IOException { // モデルパスを検証して設定するヘルパー
        if (!Files.exists(newPath)) { // ファイルが存在しない場合
            throw new IOException("Model not found at: " + newPath); // 見つからない旨の例外を投げる
        }
        this.modelPath = newPath; // パスをフィールドに保存
    }

    private synchronized void loadSession() throws OrtException { // モデルをロードしてセッションを張るメソッド
        closeSession(); // 既存のセッションがあれば一旦閉じる
        OrtSession.SessionOptions options = new OrtSession.SessionOptions(); // セッションオプションを生成
        this.session = environment.createSession(modelPath.toString(), options); // モデルパスを指定してセッションを作成
    }

    private void closeSession() { // セッションを安全に閉じるヘルパー
        if (this.session != null) { // セッションが存在するか確認
            try { // クローズ時の例外に備える
                this.session.close(); // セッションを閉じる
            } catch (OrtException ex) { // クローズ失敗時の処理
                throw new IllegalStateException("Failed to close ONNX session", ex); // ランタイム例外としてラップ
            } finally { // 例外有無に関わらず実行
                this.session = null; // セッション参照をクリア
            }
        }
    }

    public synchronized void reloadModel() throws OrtException { // モデルを再読み込みする公開メソッド
        loadSession(); // セッション作成処理を再実行
    }

    /**
     * ONNX Runtimeに入力テンソルを与えて推論を実行し、所要時間も返す。 // 単一入力推論の説明
     */
    public synchronized InferenceResult runInference(String inputName, // 入力ノード名
                                                     float[] flattenedInput, // フラット化された入力データ
                                                     long[] inputShape, // 入力テンソルの形状
                                                     String outputName) throws OrtException { // 出力ノード名
        if (session == null) { // セッション未初期化のチェック
            throw new IllegalStateException("ONNX session is not initialized"); // 不正状態を通知
        }
        long start = System.nanoTime(); // 推論開始時刻をナノ秒で取得
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(flattenedInput), inputShape); // 入力データをTensor化
             OrtSession.Result result = session.run(Map.of(inputName, inputTensor))) { // セッションに入力して推論を実行
            Optional<OnnxValue> optionalValue = result.get(outputName); // 指定出力をOptionalで取得
            OnnxValue outputValue = optionalValue.orElseThrow( // 出力が無ければ例外を投げる
                () -> new IllegalStateException("Output '" + outputName + "' not found in inference result") // 出力欠如のエラーメッセージ
            );
            if (!(outputValue instanceof OnnxTensor tensor)) { // 出力がTensor型か確認
                throw new IllegalStateException("Expected tensor output but received: " + outputValue.getClass()); // 型不一致を通知
            }
            float[] output = toFloatArray(tensor); // Tensorをfloat配列に変換
            long duration = System.nanoTime() - start; // 推論にかかった時間を計算
            return new InferenceResult(output, duration); // 出力と時間をまとめて返す
        }
    }

    /**
     * 複数入力（列ごとのTensorなど）を受け取って推論を実行する。 // 複数入力版の説明
     * 値は float[] または String[] を想定しており、内部でOnnxTensorに変換する。 // 受け付ける型の補足
     */
    public synchronized InferenceResult runInference(Map<String, Object> inputs, // 入力名→値のマップ
                                                     String outputName) throws OrtException { // 取得したい出力名
        if (session == null) { // セッションが用意されているか確認
            throw new IllegalStateException("ONNX session is not initialized"); // 未初期化時は例外
        }
        Map<String, OnnxTensor> tensorMap = new HashMap<>(); // 生成したTensorを保持するマップ
        long start = System.nanoTime(); // 推論開始時刻を取得
        try { // Tensor作成と推論をまとめて処理
            for (Map.Entry<String, Object> entry : inputs.entrySet()) { // 入力マップを走査
                tensorMap.put(entry.getKey(), createTensor(entry.getValue())); // 値をTensor化して格納
            }
            try (OrtSession.Result result = session.run(tensorMap)) { // 生成したTensor群で推論を実行
                OnnxValue outputValue = result.get(outputName).orElseThrow( // 出力取得、無ければ例外
                        () -> new IllegalStateException("Output '" + outputName + "' not found in inference result") // 出力が存在しない場合のメッセージ
                );
                if (!(outputValue instanceof OnnxTensor tensor)) { // 出力がTensorかを確認
                    throw new IllegalStateException("Expected tensor output but received: " + outputValue.getClass()); // 型不一致を報告
                }
                float[] output = toFloatArray(tensor); // Tensorをfloat配列に変換
                long duration = System.nanoTime() - start; // 推論時間を計算
                return new InferenceResult(output, duration); // 結果を返却
            }
        } finally { // Tensorのクローズを必ず実施
            for (OnnxTensor tensor : tensorMap.values()) { // 生成したTensorを走査
                if (tensor != null) { // null安全のためチェック
                    tensor.close(); // Tensorをクローズしてリソース解放
                }
            }
        }
    }

    private OnnxTensor createTensor(Object value) throws OrtException { // 値の型に応じてTensorを生成するヘルパー
        if (value instanceof float[] floats) { // float配列の場合
            return OnnxTensor.createTensor(environment, FloatBuffer.wrap(floats), new long[]{floats.length, 1}); // float Tensorを作成
        }
        if (value instanceof String[] strings) { // 文字列配列の場合
            return OnnxTensor.createTensor(environment, strings, new long[]{strings.length, 1}); // string Tensorを作成
        }
        throw new IllegalArgumentException("Unsupported input type: " + value.getClass()); // 対応外の型なら例外
    }

    private float[] toFloatArray(OnnxTensor tensor) throws OrtException { // Tensorの値をfloat配列にコピーするヘルパー
        FloatBuffer buffer = tensor.getFloatBuffer(); // Tensor内部のFloatBufferを取得
        float[] copy = new float[buffer.remaining()]; // 残り要素数に応じた配列を準備
        buffer.get(copy); // バッファの内容を配列に読み出す
        return copy; // コピーした配列を返す
    }

    @Override
    public synchronized void close() { // Closeable実装としてリソースを解放する
        closeSession(); // セッションを先に閉じる
        environment.close(); // 環境をクローズしてリソース解放
    }
} // OnnxPredictorクラスの終端
