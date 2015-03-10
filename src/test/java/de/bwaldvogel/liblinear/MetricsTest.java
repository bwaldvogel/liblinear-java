package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.F1Score.*;
import static org.fest.assertions.Assertions.assertThat;

import org.junit.Test;

/**
 * Created by christinamueller on 2/23/15.
 */
public class MetricsTest {


    @Test
    public void testEvaluateF1Weighted() {
        double[][] y = {{1.0, 2.0, 3.0, 2.0,3.0}};
        double[][] pred = {{1.0, 2.0, 3.0, 1.0,2.0}};
        double[] labels = new double[]{1.0, 2.0, 3.0};
        F1Score f1 = new F1Score(new F1Score.F1Weighted(),labels);
        Crossvalidation.Result result = f1.evaluate(y, pred);
        assertThat(result.mean - 0.6).isLessThan(0.000001);
    }

    @Test
    public void testEvaluateF1Macro() {
        double[][] y = {{1.0, 2.0, 3.0, 2.0,3.0}};
        double[][] pred = {{1.0, 2.0, 3.0, 1.0,2.0}};
        double[] labels = new double[]{1.0, 2.0, 3.0};
        F1Score f1 = new F1Score(new F1Score.F1Macro(),labels);
        Crossvalidation.Result result = f1.evaluate(y, pred);
        assertThat(result.mean - 0.61111111).isLessThan(0.000001);
    }


    @Test
    public void testAccuracy() {
        double[][] y = {{1.0, 2.0, 3.0, 2.0,3.0}};
        double[][] pred = {{1.0, 2.0, 3.0, 1.0,2.0}};
        Metrics f1 = new Accuracy();
        Crossvalidation.Result result = f1.evaluate(y, pred);
        System.out.println(result.mean);
        assertThat(result.mean == 0.6);
    }



}
