package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/24/15.
 */
public class Crossvalidation {


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
    public static void crossvalidation(Problem prob, Parameter[] param, int nr_fold, Crossvalidation.Result[] performance, Metrics metrics) {

        for (int i = 0; i < param.length; i++) {
            double[][] target = new double[nr_fold][];
            double[][] y = new double[nr_fold][];
            Linear.crossValidation(prob, param[i], nr_fold, target, y);
            performance[i] = metrics.evaluate(y, target);
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
