package de.bwaldvogel.liblinear;

import static org.fest.assertions.Assertions.assertThat;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;


public class ParameterTest {

    private Parameter _param;

    @Before
    public void setUp() {
        _param = new Parameter(SolverType.L2R_L1LOSS_SVC_DUAL, 100, 1e-3);
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

}
