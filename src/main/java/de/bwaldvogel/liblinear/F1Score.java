package de.bwaldvogel.liblinear;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by christinamueller on 2/23/15.
 */
public class F1Score implements Metrics
{
    double[] classes;
    public F1Score(double[] classes){
        this.classes = classes;
    }


    @Override
    public double evaluate(double[] trueLabels, double[] predLabels) {
        return getAvgF1(trueLabels,predLabels,classes);
    }

    private double getAvgF1(double[] trueLabels, double[] predicted, double[] labels) {

        if (trueLabels.length != predicted.length) {
            System.err.println("Number of predicted and true labels are not equal\n");
        }

        double[] f1Scores = new double[labels.length];
        double[] support = new double[labels.length];
        double avgF1 = 0.0d;
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
            avgF1 = avgF1 + support[j] * f1Scores[j];
        }
        return avgF1;
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





}





