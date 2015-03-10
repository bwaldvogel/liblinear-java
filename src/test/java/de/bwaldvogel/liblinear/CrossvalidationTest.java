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
        Crossvalidation.Result[] performance = new Crossvalidation.Result[params.length];
        Crossvalidation.crossvalidationWithMetrics(prob, params, 2, performance, new Accuracy());
        for (int i = 0; i < performance.length; i++) {
            assertThat(performance[i].mean).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(1);
        }

    }

}
