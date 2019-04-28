package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.L1R_LR;
import static de.bwaldvogel.liblinear.SolverType.L2R_L1LOSS_SVC_DUAL;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;


public class ParameterTest {

    private Parameter _param;

    @Before
    public void setUp() {
        _param = new Parameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3);
    }

    @Test
    public void testDefaults() {
        Parameter parameters = new Parameter(L1R_LR, 1, 0.1);
        assertThat(parameters.getP()).isEqualTo(0.1);
        assertThat(parameters.getMaxIters()).isEqualTo(1000);
    }

    @Test
    public void testSetWeights() {

        assertThat(_param.weight).isNull();
        assertThat(_param.getNumWeights()).isEqualTo(0);

        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        _param.setWeights(weights, weightLabels);

        assertThat(_param.getNumWeights()).isEqualTo(6);

        // assert parameter uses a copy
        weights[0]++;
        assertThat(_param.getWeights()[0]).isEqualTo(0);
        weightLabels[0]++;
        assertThat(_param.getWeightLabels()[0]).isEqualTo(1);

        weights = new double[] {0, 1, 2, 3, 4, 5};
        weightLabels = new int[] {1};
        try {
            _param.setWeights(weights, weightLabels);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("same").contains("length");
        }
    }

    @Test
    public void testGetWeights() {
        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        _param.setWeights(weights, weightLabels);

        assertThat(_param.getWeights()).isEqualTo(weights);
        _param.getWeights()[0]++; // shouldn't change the parameter as we should get a copy
        assertThat(_param.getWeights()).isEqualTo(weights);

        assertThat(_param.getWeightLabels()).isEqualTo(weightLabels);
        _param.getWeightLabels()[0]++; // shouldn't change the parameter as we should get a copy
        assertThat(_param.getWeightLabels()[0]).isEqualTo(1);
    }

    @Test
    public void testSetC() {
        _param.setC(0.0001);
        assertThat(_param.getC()).isEqualTo(0.0001);
        _param.setC(1);
        _param.setC(100);
        assertThat(_param.getC()).isEqualTo(100);
        _param.setC(Double.MAX_VALUE);

        try {
            _param.setC(-1);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("must").contains("not").contains("<= 0");
        }

        try {
            _param.setC(0);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("must").contains("not").contains("<= 0");
        }
    }

    @Test
    public void testSetEps() {
        _param.setEps(0.0001);
        assertThat(_param.getEps()).isEqualTo(0.0001);
        _param.setEps(1);
        _param.setEps(100);
        assertThat(_param.getEps()).isEqualTo(100);
        _param.setEps(Double.MAX_VALUE);

        try {
            _param.setEps(-1);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("must").contains("not").contains("<= 0");
        }

        try {
            _param.setEps(0);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("must").contains("not").contains("<= 0");
        }
    }

    @Test
    public void testSetSolverType() {
        for (SolverType type : SolverType.values()) {
            _param.setSolverType(type);
            assertThat(_param.getSolverType()).isEqualTo(type);
        }
        try {
            _param.setSolverType(null);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("must").contains("not").contains("null");
        }
    }

    @Test
    public void testClone_Simple() throws Exception {
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
    public void testClone_Full() throws Exception {
        Parameter parameter = new Parameter(L1R_LR, 123.456, 0.123, 9000, 1.2);
        parameter.setWeights(new double[] {1, 2}, new int[] {3, 4});
        Parameter clone = parameter.clone();
        assertThat(clone.getSolverType()).isEqualTo(L1R_LR);
        assertThat(clone.getC()).isEqualTo(123.456);
        assertThat(clone.getEps()).isEqualTo(0.123);
        assertThat(clone.getMaxIters()).isEqualTo(9000);
        assertThat(clone.getP()).isEqualTo(1.2);
        assertThat(clone.getWeights()).containsExactly(1, 2);
        assertThat(clone.getWeightLabels()).containsExactly(3, 4);
        assertThat(clone.getNumWeights()).isEqualTo(2);
    }

}
