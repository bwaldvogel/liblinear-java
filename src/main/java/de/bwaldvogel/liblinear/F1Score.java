package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/23/15.
 */
public class F1Score implements Metrics {
    double[] classes;
    F1Average average;

    public F1Score(F1Average average, double[] classes) {
        this.average = average;
        this.classes = classes;
    }

    @Override
    public Crossvalidation.Result evaluate(double[][] trueLabels, double[][] predLabels) {
        int noFold = trueLabels.length;
        double[] acc = new double[noFold];
        double[][] f1PerClass = new double[noFold][];
        double[][] precisionPerClass = new double[noFold][];
        double[][] recallPerClass = new double[noFold][];
        for (int i = 0; i < noFold; i++) {
            F1AverageResult result = getAverageOverClass(trueLabels[i], predLabels[i], classes);
            acc[i] = result.getAverage();
            f1PerClass[i] = result.getF1Scores();
            precisionPerClass[i] = result.getPrecision();
            recallPerClass[i] = result.getRecall();
        }
        double mean = Crossvalidation.mean(acc);
        double std = Crossvalidation.std(mean, acc);
        return new Crossvalidation.Result(mean, std, f1PerClass, precisionPerClass, recallPerClass);


    }

    private F1AverageResult getAverageOverClass(double[] trueLabels, double[] predicted, double[] labels) {

        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }

        double[] f1Scores = new double[labels.length];
        double[] precision = new double[labels.length];
        double[] recall = new double[labels.length];
        double[] support = new double[labels.length];

        for (int j = 0; j < labels.length; j++) {

            double[] trueConverted = new double[predicted.length];
            double[] predConverted = new double[predicted.length];
            int labelCounter = 0;
            double label = labels[j];
            for (int i = 0; i < predicted.length; i++) {
                if (trueLabels[i] == label) {
                    trueConverted[i] = 1;
                    labelCounter++;
                } else
                    trueConverted[i] = -1;
                if (predicted[i] == label) predConverted[i] = 1;
                else
                    predConverted[i] = -1;
            }

            support[j] = (double) labelCounter / trueLabels.length;
            double[] result = getF1(trueConverted, predConverted);
            f1Scores[j] = result[0];
            precision[j] = result[1];
            recall[j] = result[2];
        }
        return new F1AverageResult(average.getAverage(f1Scores, support), f1Scores, precision, recall);
    }

    private double[] getF1(double[] trueLabels, double[] predicted) {

        int tp = 0, fn = 0, fp = 0;
        double precision, recall;
        double fscore;

        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }
        for (int i = 0; i < trueLabels.length; ++i) {

            if (trueLabels[i] == 1 && predicted[i] == 1)
                tp++;
            else if (trueLabels[i] == -1 && predicted[i] == 1)
                fp++;
            else if (trueLabels[i] == 1 && predicted[i] == -1) fn++;

        }

        if (tp + fp == 0) {
            System.out.println("Warning: No positive predicted label.\n");
            precision = 0;
        } else
            precision = tp / (double) (tp + fp);
        if (tp + fn == 0) {
            System.out.println("Warning: No positive true label.\n");
            recall = 0;
        } else
            recall = tp / (double) (tp + fn);

        if (precision + recall == 0) {
            System.out.println("Warning: precision + recall = 0.\n");
            fscore = 0;
        } else
            fscore = 2 * precision * recall / (precision + recall);

        double[] result = new double[3];
        result[0] = fscore;
        result[1] = precision;
        result[2] = recall;
        return result;

    }

    private double[] getMean(double[][] f1PerClass) {
        if (f1PerClass.length > 0) {

            int noClass = f1PerClass[0].length;
            int noFolds = f1PerClass.length;
            double[] avgPerClass = new double[noClass];

            for (int j = 0; j < noClass; j++) {
                double[] scores = new double[noFolds];
                for (int i = 0; i < noFolds; i++) {
                    scores[i] = f1PerClass[i][j];
                }
                avgPerClass[j] = Crossvalidation.mean(scores);
            }
            return avgPerClass;
        }
        return new double[0];
    }

    public static class F1Macro implements F1Average {

        @Override
        public double getAverage(double[] scores, double[] support) {
            return Crossvalidation.mean(scores);
        }

    }

    public static class F1Weighted implements F1Average {

        @Override
        public double getAverage(double[] scores, double[] support) {
            double avgF1 = 0.0d;
            for (int i = 0; i < support.length; i++) {
                avgF1 = avgF1 + support[i] * scores[i];
            }
            return avgF1;
        }

    }

}









