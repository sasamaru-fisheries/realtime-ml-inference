package com.example.pmml; // pmmlパッケージ配下のPMML推論クラス

import org.jpmml.evaluator.Computable; // PMMLの計算可能な値を扱うためのインターフェース
import org.jpmml.evaluator.Evaluator; // PMMLモデルの評価を行うクラス
import org.dmg.pmml.FieldName; // PMMLのフィールド名を表すクラス
import org.jpmml.evaluator.InputField; // PMML入力フィールドを表すクラス
import org.jpmml.evaluator.ModelEvaluatorBuilder; // Evaluatorを組み立てるビルダー
import org.jpmml.evaluator.ProbabilityDistribution; // 確率分布を扱うクラス
import org.jpmml.evaluator.TargetField; // 予測ターゲットフィールドを表すクラス
import org.jpmml.model.PMMLUtil; // PMMLの読み込みユーティリティ

import java.io.Closeable; // リソースクローズを扱うインターフェース
import java.io.FileInputStream; // ファイル入力ストリームを扱うためのクラス
import java.io.IOException; // 入出力例外を扱うためのインポート
import java.io.InputStream; // 汎用入力ストリームのインポート
import java.nio.file.Files; // ファイル存在確認などのユーティリティ
import java.nio.file.Path; // ファイルパスを表すPathクラス
import java.nio.file.Paths; // 文字列からPathを生成するユーティリティ
import java.util.HashMap; // ハッシュマップ実装
import java.util.List; // Listインターフェース
import java.util.Map; // Mapインターフェース

/**
 * PMMLモデルを読み込み、評価・再読み込みを行うヘルパークラス。 // クラスの役割を説明
 */
public class PmmlPredictor implements Closeable { // PMML推論を担当するクラス
    private Path modelPath; // PMMLモデルファイルのパスを保持
    private Evaluator evaluator; // モデル評価を行うEvaluatorインスタンス

    public PmmlPredictor(String modelPath) throws IOException { // コンストラクタでモデルパスを受け取る
        setModelPathInternal(Paths.get(modelPath)); // パス設定と存在確認を行う
        loadEvaluator(); // モデルを読み込みEvaluatorを用意
    }

    /**
     * 使用するPMMLファイルを切り替える。reloadModel()で再読み込みしてください。 // モデル差し替えの説明
     */
    public synchronized void setModelPath(String newModelPath) throws IOException { // モデルパスを変更する公開メソッド
        setModelPathInternal(Paths.get(newModelPath)); // 文字列パスをPathに変換して設定
    }

    private void setModelPathInternal(Path newPath) throws IOException { // モデルパスを内部的に設定する
        if (!Files.exists(newPath)) { // ファイルが存在するか確認
            throw new IOException("PMML model not found at: " + newPath); // 見つからない場合は例外
        }
        this.modelPath = newPath; // パスをフィールドに保存
    }

    private synchronized void loadEvaluator() throws IOException { // PMMLモデルを読み込みEvaluatorを生成
        try (InputStream is = new FileInputStream(modelPath.toFile())) { // ファイル入力ストリームを開く
            this.evaluator = new ModelEvaluatorBuilder(PMMLUtil.unmarshal(is)).build(); // PMMLをデシリアライズしてEvaluatorを構築
            this.evaluator.verify(); // モデルが利用可能か検証
        } catch (Exception ex) { // 読み込みやビルドで例外が出た場合
            throw new IOException("Failed to load PMML model", ex); // IOExceptionにラップして再送出
        }
    }

    public synchronized void reloadModel() throws IOException { // モデルを再読み込みする公開メソッド
        loadEvaluator(); // Evaluator生成処理を再実行
    }

    /**
     * 入力フィールド名と値のマップを受け取り、PMMLモデルで推論する。 // 推論の概要説明
     * 返却値には予測ラベルと各クラス確率、実行時間(ナノ秒)が含まれる。 // 戻り値に含まれる情報を説明
     */
    public synchronized InferenceResult runInference(Map<String, ?> inputValues) throws IOException { // 推論を実行するメソッド
        if (evaluator == null) { // Evaluatorが初期化されているか確認
            throw new IllegalStateException("Evaluator is not initialized"); // 未初期化なら例外
        }
        long start = System.nanoTime(); // 推論開始時刻をナノ秒で取得

        Map<FieldName, Object> arguments = new HashMap<>(); // PMML入力用の引数マップを初期化
        List<InputField> inputFields = evaluator.getInputFields(); // モデルの入力フィールド一覧を取得
        for (InputField inputField : inputFields) { // 各入力フィールドを処理
            FieldName fieldName = inputField.getName(); // PMML上のフィールド名を取得
            String name = fieldName.getValue(); // 生の名前文字列を取得
            Object rawValue = inputValues.get(name); // 入力マップから元の値を取得
            Object pmmlValue = inputField.prepare(rawValue); // PMMLが期待する型へ変換
            arguments.put(fieldName, pmmlValue); // 変換後の値を引数マップに登録
        }

        Map<FieldName, ?> results = evaluator.evaluate(arguments); // 引数を渡してモデルを評価する

        List<TargetField> targetFields = evaluator.getTargetFields(); // ターゲットフィールド一覧を取得
        if (targetFields.isEmpty()) { // ターゲットが無い場合
            throw new IllegalStateException("No target field found in PMML model"); // モデル不備を通知
        }
        FieldName targetName = targetFields.get(0).getName(); // 最初のターゲットフィールド名を取得
        Object targetValue = results.get(targetName); // 推論結果からターゲット値を取得
        String predictedLabel = extractLabel(targetValue); // 予測ラベルを文字列化

        Map<String, Double> probabilities = new HashMap<>(); // クラス確率を格納するマップ
        if (targetValue instanceof ProbabilityDistribution distribution) { // ターゲット値が確率分布の場合
            for (Object key : distribution.getCategories()) { // 分布が持つカテゴリを走査
                probabilities.put(String.valueOf(key), distribution.getProbability(key)); // 確率をマップに格納
            }
        } else { // ターゲット値に分布が無い場合
            for (Object value : results.values()) { // 代わりに結果マップ内を走査
                if (value instanceof ProbabilityDistribution distribution2) { // 確率分布を見つけたら
                    for (Object key : distribution2.getCategories()) { // カテゴリを走査
                        probabilities.put(String.valueOf(key), distribution2.getProbability(key)); // 確率を格納
                    }
                    break; // 分布を見つけたら探索終了
                }
            }
        }

        long duration = System.nanoTime() - start; // 推論に要した時間を算出
        return new InferenceResult(predictedLabel, probabilities, duration); // ラベル・確率・時間をまとめて返却
    }

    private String extractLabel(Object targetValue) { // ターゲット値からラベル文字列を取り出すヘルパー
        if (targetValue instanceof Computable computable) { // Computable型の場合
            Object computed = computable.getResult(); // 実際の値を取得
            return String.valueOf(computed); // 文字列化して返す
        }
        return String.valueOf(targetValue); // その他の場合はそのまま文字列化
    }

    @Override
    public synchronized void close() { // Closeable実装としてのクローズ処理
        this.evaluator = null; // Evaluatorはclose不要だが参照をクリアしておく
    }
} // PmmlPredictorクラスの終端
