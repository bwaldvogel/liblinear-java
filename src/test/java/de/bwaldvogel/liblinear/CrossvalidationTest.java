package de.bwaldvogel.liblinear;


import org.junit.Before;
import org.junit.Test;


import static org.fest.assertions.Assertions.assertThat;

/**
 * Created by christinamueller on 2/24/15.
 */
public class CrossvalidationTest {
    Problem prob;
    Parameter[] params;

    @Before
    public void setUp() {
        prob = LinearTest.createRandomProblem(3);
        double[] C = {0.1, 1.0, 10.0};
        params = new Parameter[3];
        for (int i = 0; i < C.length; i++) {
            params[i] = new Parameter(SolverType.L2R_LR, C[i], 0.01);
        }
    }

    @Test
    public void testCrossValidate() {
        double[] performance = new double[params.length];
        Crossvalidation.crossvalidation(prob, params, 2, performance, new Accuracy());
        for (int i = 0; i < performance.length; i++) {
            assertThat(performance[i]).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(1);
        }

    }

    @Test
    public void testCrossValidateF1() {
        double[] performance = new double[params.length];
        double[] labels = {1, 2, 3};
        Crossvalidation.crossvalidation(prob, params, 2, performance, new F1Score(labels));
        for (int i = 0; i < performance.length; i++) {
            assertThat(performance[i]).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(1);
        }

    }


}
