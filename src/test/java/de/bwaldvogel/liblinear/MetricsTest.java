package de.bwaldvogel.liblinear;

import static org.fest.assertions.Assertions.assertThat;

import org.junit.Test;

/**
 * Created by christinamueller on 2/23/15.
 */
public class MetricsTest {


    @Test
    public void testGetAvgF1Score() {
        double[] trueLabels = new double[]{1.0, 2.0, 3.0, 2.0,3.0};
        double[] predLabels = new double[]{1.0, 2.0, 3.0, 1.0,2.0};
        double[] labels = new double[]{1.0, 2.0, 3.0};
        Metrics f1 = new F1Score(labels);
        double result = f1.evaluate(trueLabels, predLabels);
        assertThat(result- 0.6).isLessThan(0.000001);
    }


}
