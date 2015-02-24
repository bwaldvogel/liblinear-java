package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/24/15.
 */
public class Accuracy implements Metrics {


    @Override
    public double evaluate(double[] trueLabels, double[] predLabels) {
        return getAccuracy(trueLabels,predLabels);
    }

    private double getAccuracy(double[] trueLabels, double[] predicted) {
        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }
        int total_correct = 0;
        for (int i = 0; i < trueLabels.length; i++)
            if (trueLabels[i] == predicted[i]) ++total_correct;
        return total_correct/(double)trueLabels.length;

    }

}
