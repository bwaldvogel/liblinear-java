package de.bwaldvogel.liblinear;

import java.util.Random;
/**
 * Created by christinamueller on 2/24/15.
 */

public class Crossvalidation {

    static Random              random              = new Random(1);

    public static class Result {
        double mean;
        double std;
        Result(double mean, double std) {
            this.mean = mean;
            this.std = std;
        }

        public double getMean() {
            return mean;
        }

        public double getStd() {
            return std;
        }
    }


    public static void crossvalidation(Problem prob, Parameter param, int nr_fold, double[] target) {
        Linear.crossValidation(prob, param, nr_fold, target);
    }

    /**
     * Cross-validation using Metrics for evaluation.
     *
     * @param performance holds the evaluation values for each parameter
     * @param metrics Defines the metrics by which the predicted labels on the validation set should be evaluated
     */
    public static void crossvalidationWithMetrics(Problem prob, Parameter[] params, int nr_fold, Crossvalidation.Result[] performance, Metrics metrics) {

        for (int i = 0; i < params.length; i++) {
            double[][] target = new double[nr_fold][];
            double[][] perm_y = new double[nr_fold][];
            cv(prob, params[i], nr_fold, target, perm_y);
            performance[i] = metrics.evaluate(perm_y, target);
        }
    }

    private static void cv(Problem prob, Parameter param, int nr_fold, double[][] target, double[][] perm_y) {

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
        for (i = 0; i < l; i++) {
            int j = i + random.nextInt(l - i);
            Linear.swap(perm, i, j);
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


    public static double mean(double[] values){
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum / values.length;
    }

    public static double std(double mean, double[] values){
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.pow(values[i] - mean, 2);
        }
        return Math.sqrt(sum / values.length);
    }



}
