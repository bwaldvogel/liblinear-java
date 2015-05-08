package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/24/15.
 */
public class Accuracy implements Metrics {


    @Override
    public Crossvalidation.Result evaluate(double[][] trueLabels, double[][] predLabels) {
        int noFold = trueLabels.length;
        double[] acc = new double[noFold];
        for (int i = 0; i < noFold; i++) {
            acc[i] = getAccuracy(trueLabels[i], predLabels[i]);
        }
        double mean = Crossvalidation.mean(acc);
        double std = Crossvalidation.std(mean, acc);
        return new Crossvalidation.Result(mean, std, new double[0][0], new double[0][0], new double[0][0]);
    }

    private double getAccuracy(double[] trueLabels, double[] predicted) {
        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }
        int total_correct = 0;
        for (int i = 0; i < trueLabels.length; i++)
            if (trueLabels[i] == predicted[i]) ++total_correct;
        return total_correct / (double) trueLabels.length;

    }


}
