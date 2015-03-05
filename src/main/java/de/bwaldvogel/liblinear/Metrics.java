package de.bwaldvogel.liblinear;

/**
 * Created by christinamueller on 2/24/15.
 */
public interface Metrics {

    public Crossvalidation.Result evaluate(double[][] trueLabels, double[][] predLabels);

}
