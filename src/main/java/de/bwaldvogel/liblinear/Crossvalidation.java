package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/24/15.
 */
public class Crossvalidation {


    public static void crossvalidation(Problem prob, Parameter param, int nr_fold, double[] target) {
        Linear.crossValidation(prob, param, nr_fold, target);
    }

    /**
     * Cross-validation using Metrics for evaluation.
     *
     * @param performance holds the evaluation values for each parameter
     * @param metrics Defines the metrics by which the predicted labels on the validation set should be evaluated
     */
    public static void crossvalidation(Problem prob, Parameter[] param, int nr_fold, double[] performance, Metrics metrics) {

        for (int i = 0; i < param.length; i++) {
            double[] target = new double[prob.l];
            Linear.crossValidation(prob, param[i], nr_fold, target);
            performance[i] = metrics.evaluate(prob.y, target);
        }
    }

    public static int argmax(double[] values) {
        if (values.length == 0) System.err.println("Cannot get an argmax of an empty array");
        int maxInd = 0;
        double max = values[0];
        for (int i = 0; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                maxInd = i;
            }
        }
        return maxInd;

    }


}
