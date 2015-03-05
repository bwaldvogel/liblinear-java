package de.bwaldvogel.liblinear;

import java.io.PrintStream;

/**
 * Created by christinamueller on 2/23/15.
 */
public class F1Score implements Metrics
{
    double[] classes;
    F1Average average;

    public F1Score(double[] classes, F1Average average){
        this.classes = classes;
        this.average = average;
    }


    @Override
    public Crossvalidation.Result evaluate(double[][] trueLabels, double[][] predLabels) {
        int noFold = trueLabels.length;
        double[] acc = new double[noFold];
        for (int i = 0; i < noFold; i++) {
            acc[i]= getAvgF1(trueLabels[i],predLabels[i],classes);
        }
        double mean = Crossvalidation.mean(acc);
        double std = Crossvalidation.std(mean,acc);
        return new Crossvalidation.Result(mean,std);


    }

    private double getAvgF1(double[] trueLabels, double[] predicted, double[] labels) {

        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }

        double[] f1Scores = new double[labels.length];
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
            f1Scores[j] = getF1(trueConverted, predConverted);
        }
        return average.getAverage(f1Scores,support);
    }

    private double getF1(double[] trueLabels, double[] predicted) {

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


        return fscore;

    }

    public static class F1Macro implements F1Average {

        @Override
        public double getAverage(double[] scores, double[] support) {
            return average(scores);
        }

        private double average(double[] values){
            return Crossvalidation.mean(values);
        }

    }

    public static class F1Weighted implements F1Average {

        @Override
        public double getAverage(double[] scores, double[] support) {
            return average(scores,support);
        }
        private double average(double[] values,double[] support){
            double avgF1 = 0.0d;
            for (int i = 0; i < support.length; i++) {
                avgF1 = avgF1 + support[i] * values[i];
            }
            return avgF1;
        }

    }


}





