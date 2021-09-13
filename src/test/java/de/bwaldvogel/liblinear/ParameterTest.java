package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.*;
import static org.assertj.core.api.Assertions.*;

import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;


class ParameterTest {

    private Parameter param;

    @BeforeEach
    public void setUp() {
        param = new Parameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3);
    }

    @Test
    void testDefaults() {
        Parameter parameters = new Parameter(L1R_LR, 1, 0.1);
        assertThat(parameters.getP()).isEqualTo(0.1);
        assertThat(parameters.getMaxIters()).isEqualTo(1000);
    }

    @Test
    void testSetWeights() {
        assertThat(param.weight).isNull();
        assertThat(param.getNumWeights()).isEqualTo(0);

        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        param.setWeights(weights, weightLabels);

        assertThat(param.getNumWeights()).isEqualTo(6);

        // assert parameter uses a copy
        weights[0]++;
        assertThat(param.getWeights()[0]).isEqualTo(0);
        weightLabels[0]++;
        assertThat(param.getWeightLabels()[0]).isEqualTo(1);
    }

    @Test
    void testSetWeights_IllegalArgument() {
        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1};

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> param.setWeights(weights, weightLabels))
            .withMessageContaining("same", "length");
    }

    @Test
    void testGetWeights() {
        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        param.setWeights(weights, weightLabels);

        assertThat(param.getWeights()).isEqualTo(weights);
        param.getWeights()[0]++; // shouldn't change the parameter as we should get a copy
        assertThat(param.getWeights()).isEqualTo(weights);

        assertThat(param.getWeightLabels()).isEqualTo(weightLabels);
        param.getWeightLabels()[0]++; // shouldn't change the parameter as we should get a copy
        assertThat(param.getWeightLabels()[0]).isEqualTo(1);
    }

    @Test
    void testSetC() {
        param.setC(0.0001);
        assertThat(param.getC()).isEqualTo(0.0001);
        param.setC(1);
        param.setC(100);
        assertThat(param.getC()).isEqualTo(100);
        param.setC(Double.MAX_VALUE);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> param.setC(-1))
            .withMessageContainingAll("must", "not", "<= 0");

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> param.setC(0))
            .withMessageContainingAll("must", "not", "<= 0");
    }

    @Test
    void testSetEps() {
        param.setEps(0.0001);
        assertThat(param.getEps()).isEqualTo(0.0001);
        param.setEps(1);
        param.setEps(100);
        assertThat(param.getEps()).isEqualTo(100);
        param.setEps(Double.MAX_VALUE);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> param.setEps(-1))
            .withMessageContainingAll("must", "not", "<= 0");

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> param.setEps(0))
            .withMessageContainingAll("must", "not", "<= 0");
    }

    @Test
    void testSetSolverType() {
        for (SolverType type : SolverType.values()) {
            param.setSolverType(type);
            assertThat(param.getSolverType()).isEqualTo(type);
        }

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setSolverType(null))
            .withMessageContainingAll("must", "not", "null");
    }

    @Test
    void testSetInitSol() {
        assertThat(param.init_sol).isNull();

        double[] init_sol = new double[] {0, 1, 2, 3, 4, 5};
        param.setInitSol(init_sol);

        // assert parameter uses a copy
        init_sol[0]++;
        assertThat(param.getInitSol()[0]).isEqualTo(0);
    }

    @Test
    void testSetP() throws Exception {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setP(-1))
            .withMessage("p must not be less than 0");
    }

    @Test
    void testSetNu() throws Exception {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setNu(-0.1))
            .withMessage("nu must not be <=0");

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setNu(0.0))
            .withMessage("nu must not be <=0");

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setNu(1.0))
            .withMessage("nu must not be >=1");

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() ->  param.setNu(1.5))
            .withMessage("nu must not be >=1");
    }

    @Test
    void testGetInitSol() {
        assertThat(param.getInitSol()).isNull();

        double[] init_sol = new double[] {0, 1, 2, 3, 4, 5};
        param.setInitSol(init_sol);

        assertThat(param.getInitSol()).isNotNull();
        assertThat(param.getInitSol()).isEqualTo(init_sol);
        param.getInitSol()[0]++; // shouldn't change the parameter as we should get a copy
        assertThat(param.getInitSol()).isEqualTo(init_sol);
    }

    @Test
    void testClone_Simple() throws Exception {
        Parameter parameter = new Parameter(L1R_LR, 123.456, 0.123);
        Parameter clone = parameter.clone();
        assertThat(clone.getSolverType()).isEqualTo(L1R_LR);
        assertThat(clone.getC()).isEqualTo(123.456);
        assertThat(clone.getEps()).isEqualTo(0.123);
        assertThat(clone.getWeights()).isNull();
        assertThat(clone.getWeightLabels()).isNull();
        assertThat(clone.getNumWeights()).isEqualTo(0);
    }

    @Test
    void testClone_Full() throws Exception {
        Parameter parameter = new Parameter(L1R_LR, 123.456, 0.123, 9000, 1.2);
        parameter.setWeights(new double[] {1, 2}, new int[] {3, 4});
        Random random = new Random(123);
        parameter.setRandom(random);
        Parameter clone = parameter.clone();
        assertThat(clone.getSolverType()).isEqualTo(L1R_LR);
        assertThat(clone.getC()).isEqualTo(123.456);
        assertThat(clone.getEps()).isEqualTo(0.123);
        assertThat(clone.getMaxIters()).isEqualTo(9000);
        assertThat(clone.getP()).isEqualTo(1.2);
        assertThat(clone.getWeights()).containsExactly(1, 2);
        assertThat(clone.getWeightLabels()).containsExactly(3, 4);
        assertThat(clone.getNumWeights()).isEqualTo(2);

        assertThat(clone.random).isNotSameAs(random);
        assertThat(random.nextInt()).isEqualTo(clone.random.nextInt());
    }

}
