package com.example.pmml;

import org.jpmml.evaluator.Computable;
import org.jpmml.evaluator.Evaluator;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.ProbabilityDistribution;
import org.jpmml.evaluator.TargetField;
import org.jpmml.model.PMMLUtil;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PMMLモデルを読み込み、評価・再読み込みを行うヘルパークラス。
 */
public class PmmlPredictor implements Closeable {
    private Path modelPath;
    private Evaluator evaluator;

    public PmmlPredictor(String modelPath) throws IOException {
        setModelPathInternal(Paths.get(modelPath));
        loadEvaluator();
    }

    /**
     * 使用するPMMLファイルを切り替える。reloadModel()で再読み込みしてください。
     */
    public synchronized void setModelPath(String newModelPath) throws IOException {
        setModelPathInternal(Paths.get(newModelPath));
    }

    private void setModelPathInternal(Path newPath) throws IOException {
        if (!Files.exists(newPath)) {
            throw new IOException("PMML model not found at: " + newPath);
        }
        this.modelPath = newPath;
    }

    private synchronized void loadEvaluator() throws IOException {
        try (InputStream is = new FileInputStream(modelPath.toFile())) {
            this.evaluator = new ModelEvaluatorBuilder(PMMLUtil.unmarshal(is)).build();
            this.evaluator.verify();
        } catch (Exception ex) {
            throw new IOException("Failed to load PMML model", ex);
        }
    }

    public synchronized void reloadModel() throws IOException {
        loadEvaluator();
    }

    /**
     * 入力フィールド名と値のマップを受け取り、PMMLモデルで推論する。
     * 返却値には予測ラベルと各クラス確率、実行時間(ナノ秒)が含まれる。
     */
    public synchronized InferenceResult runInference(Map<String, ?> inputValues) throws IOException {
        if (evaluator == null) {
            throw new IllegalStateException("Evaluator is not initialized");
        }
        long start = System.nanoTime();

        // 入力フィールドをPMMLの型に合わせて変換する
        Map<FieldName, Object> arguments = new HashMap<>();
        List<InputField> inputFields = evaluator.getInputFields();
        for (InputField inputField : inputFields) {
            FieldName fieldName = inputField.getName();
            String name = fieldName.getValue();
            Object rawValue = inputValues.get(name);
            Object pmmlValue = inputField.prepare(rawValue);
            arguments.put(fieldName, pmmlValue);
        }

        // 推論実行
        Map<FieldName, ?> results = evaluator.evaluate(arguments);

        // ターゲット（予測ラベル）を取り出す
        List<TargetField> targetFields = evaluator.getTargetFields();
        if (targetFields.isEmpty()) {
            throw new IllegalStateException("No target field found in PMML model");
        }
        FieldName targetName = targetFields.get(0).getName();
        Object targetValue = results.get(targetName);
        String predictedLabel = extractLabel(targetValue);

        // 確率分布を取り出す（存在する場合）。ProbabilityDistributionはMapを実装しているのでコピーしてから走査する。
        Map<String, Double> probabilities = new HashMap<>();
        if (targetValue instanceof ProbabilityDistribution distribution) {
            for (Object key : distribution.getCategories()) {
                probabilities.put(String.valueOf(key), distribution.getProbability(key));
            }
        } else {
            // もしくはresults側にProbabilityDistribution型が含まれるケースもあるので走査する
            for (Object value : results.values()) {
                if (value instanceof ProbabilityDistribution distribution2) {
                    for (Object key : distribution2.getCategories()) {
                        probabilities.put(String.valueOf(key), distribution2.getProbability(key));
                    }
                    break;
                }
            }
        }

        long duration = System.nanoTime() - start;
        return new InferenceResult(predictedLabel, probabilities, duration);
    }

    private String extractLabel(Object targetValue) {
        if (targetValue instanceof Computable computable) {
            Object computed = computable.getResult();
            return String.valueOf(computed);
        }
        return String.valueOf(targetValue);
    }

    @Override
    public synchronized void close() {
        // Evaluatorはclose不要だが、将来の拡張に備えてnullクリアしておく
        this.evaluator = null;
    }
}
