package de.bwaldvogel.liblinear;

import java.util.*;

/**
 * Created by christinamueller on 2/24/15.
 */

public class Crossvalidation {

    static Random random = new Random(1);

    public static class Result {
        double mean;
        double std;
        double[][] f1PerClass;
        double[][] precisionPerClass;
        double[][] recallPerClass;


        Result(double mean, double std, double[][] f1PerClass, double[][] precisionPerClass, double[][] recallPerClass) {
            this.mean = mean;
            this.std = std;
            this.f1PerClass = f1PerClass;
            this.precisionPerClass = precisionPerClass;
            this.recallPerClass = recallPerClass;
        }

        public double getMean() {
            return mean;
        }

        public double getStd() {
            return std;
        }

        public double[][] getF1PerClass() {
            return f1PerClass;
        }
        public double[][] getPrecisionPerClass() {
            return precisionPerClass;
        }
        public double[][] getRecallPerClass() {
            return recallPerClass;
        }
    }


    public static void crossvalidation(Problem prob, Parameter param, int nr_fold, double[] target) {
        Linear.crossValidation(prob, param, nr_fold, target);
    }

    /**
     * Cross-validation using Metrics for evaluation for unbalanced data sets using weights and for each class.
     *
     * @param performance holds the evaluation values for each parameter
     * @param metrics     Defines the metrics by which the predicted labels on the validation set should be evaluated
     */
    public static void crossvalidation(Problem prob, Parameter[] params, int nr_fold, Crossvalidation.Result[] performance, Metrics metrics, boolean shuffle) {

        for (int i = 0; i < params.length; i++) {
            double[][] target = new double[nr_fold][];
            double[][] perm_y = new double[nr_fold][];
            cv(prob, params[i], nr_fold, target, perm_y,shuffle);
            performance[i] = metrics.evaluate(perm_y, target);
        }
    }


    /**
     * Cross-validation using Metrics for evaluation for unbalanced data sets using weights and for each class.
     *
     * @param performance holds the evaluation values for each parameter
     * @param metrics     Defines the metrics by which the predicted labels on the validation set should be evaluated
     */
    public static void crossvalidation(Problem prob, Parameter[] params, int nr_fold, Crossvalidation.Result[] performance, Metrics metrics) {
        crossvalidation(prob,params,nr_fold,performance,metrics,true);
    }


    private static void cv(Problem prob, Parameter param, int nr_fold, double[][] target, double[][] perm_y, boolean shullfe) {

        int i;
        int l = prob.l;
        int[] perm = new int[l];

        if (nr_fold > l) {
            nr_fold = l;
            System.err.println("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)");
        }
        int[] fold_start = new int[nr_fold + 1];

        for (i = 0; i < l; i++)
            perm[i] = i;

        if (shullfe){
            for (i = 0; i < l; i++) {
                int j = i + random.nextInt(l - i);
                Linear.swap(perm, i, j);
            }
        }

        for (i = 0; i <= nr_fold; i++)
            fold_start[i] = i * l / nr_fold;

        for (i = 0; i < nr_fold; i++) {
            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int j, k;
            Problem subprob = new Problem();

            subprob.bias = prob.bias;
            subprob.n = prob.n;
            subprob.l = l - (end - begin);
            subprob.x = new Feature[subprob.l][];
            subprob.y = new double[subprob.l];

            k = 0;
            for (j = 0; j < begin; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for (j = end; j < l; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }

            int[] classes = getClasses(prob.y);
            double[] weights = getWeights(subprob.y, classes);

            param.setWeights(weights, classes);
            Model submodel = Linear.train(subprob, param);

            double[] pred = new double[end - begin];
            double[] y = new double[end - begin];

            for (j = begin; j < end; j++) {
                pred[j - begin] = Linear.predict(submodel, prob.x[perm[j]]);
                y[j - begin] = prob.y[perm[j]];
            }
            target[i] = pred;
            perm_y[i] = y;
        }

    }

    public static double[] getWeights(double[] trainY, int[] classes) {
        Map<Integer, Double> counter = new HashMap<Integer, Double>();

        for (int i = 0; i < classes.length; i++) {
            counter.put(classes[i], 0.0);
        }
        for (int i = 0; i < trainY.length; i++) {
            int key = (int) trainY[i];
            counter.put(key, counter.get(key) + 1);
        }
        double[] weights = new double[classes.length];

        for (int i = 0; i < classes.length; i++) {
            weights[i] = (1 / counter.get(classes[i]));
        }
        return weights;
    }

    public static int[] getClasses(double[] trainY) {

        List<Double> tmp = new ArrayList<Double>(trainY.length);
        for (int i = 0; i < trainY.length; i++) {
            tmp.add(trainY[i]);
        }
        Set<Double> set = new LinkedHashSet<Double>(tmp);
        int[] unique = new int[set.size()];

        Object[] array = set.toArray();
        for (int i = 0; i < array.length; i++) {
            unique[i] = ((Double) array[i]).intValue();
        }
        return unique;
    }


    public static double mean(double[] values) {
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum / values.length;
    }

    public static double std(double mean, double[] values) {
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.pow(values[i] - mean, 2);
        }
        return Math.sqrt(sum / values.length);
    }


}
