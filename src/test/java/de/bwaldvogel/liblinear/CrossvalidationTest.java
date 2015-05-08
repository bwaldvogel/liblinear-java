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
        Crossvalidation.crossvalidation(prob, params, 2, performance, new Accuracy());
        for (int i = 0; i < performance.length; i++) {
            assertThat(performance[i].mean).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(1);
        }

    }


    @Test
    public void testGetClasses() {
        double[] classes = {1.0,4.0,2.0,1.0,2.0};
        int[] unique = Crossvalidation.getClasses(classes);
        assertThat(unique.length == 3);
        assertThat(unique[0] == 1);
        assertThat(unique[1] == 4);
        assertThat(unique[2] == 2);
    }

    @Test
    public void testGetWeigths() {
        double[] trainY = {1.0,2.0,2.0,3.0,3.0,3.0};
        int[] labels = {1,2,3};
        double[] weights = Crossvalidation.getWeights(trainY, labels);
        assertThat(weights.length == 3);
        assertThat(weights[0]== 1.0);
        assertThat(weights[1] == 0.5);
        assertThat(weights[2] == 0.3333333);

    }



}
