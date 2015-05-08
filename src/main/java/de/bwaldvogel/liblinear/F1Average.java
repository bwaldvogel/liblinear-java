package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 3/5/15.
 */
public interface F1Average {

    public double getAverage(double[] scores, double[] support);

}


class F1AverageResult {

    double average;
    double[] f1Scores;
    double[] precision;
    double[] recall;


    F1AverageResult(double average, double[] f1Scores, double[] precision, double[] recall) {
        this.average = average;
        this.f1Scores = f1Scores;
        this.precision = precision;
        this.recall = recall;
    }


    public double getAverage() {
        return average;
    }

    public double[] getF1Scores() {
        return f1Scores;
    }

    public double[] getRecall() {
        return recall;
    }

    public double[] getPrecision() {
        return precision;
    }


}


