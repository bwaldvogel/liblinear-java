package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.Test;


public class SparseOperatorTest {

    @Test
    public void testNrm2Sq() throws Exception {
        assertThat(SparseOperator.nrm2_sq(features())).isZero();
        assertThat(SparseOperator.nrm2_sq(features(1.0))).isEqualTo(1.0);
        assertThat(SparseOperator.nrm2_sq(features(1.0, 2.0))).isEqualTo(1 + 4);
        assertThat(SparseOperator.nrm2_sq(features(1.0, 2.0, 3.0))).isEqualTo(5 + 9);
    }

    @Test
    public void testDot() throws Exception {
        assertThat(SparseOperator.dot(new double[0], features())).isEqualTo(0.0);
        assertThat(SparseOperator.dot(new double[] {1.0}, features(1.0))).isEqualTo(1.0);
        assertThat(SparseOperator.dot(new double[] {1.0, 2.0}, features(1.0, 2.0))).isEqualTo(1 + 2 * 2);
        assertThat(SparseOperator.dot(new double[] {3.0, 2.0, 1.0}, features(1.0, 2.0, 3.0))).isEqualTo(3 * 1 + 2 * 2 + 1 * 3);
    }

    @Test
    public void testAxpy() throws Exception {
        assertThat(axpy(1.0, features())).isEmpty();
        assertThat(axpy(1.5, features(2.0))).containsExactly(1.5 * 2.0);

        double[] y = new double[] {1.0, 2.0};
        SparseOperator.axpy(1.5, features(2.0, 3.0), y);
        assertThat(y).containsExactly(1.0 + 1.5 * 2.0, 2.0 + 1.5 * 3.0);
    }

    private static double[] axpy(double a, Feature[] features) {
        double[] result = new double[features.length];
        SparseOperator.axpy(a, features, result);
        return result;
    }

    private static Feature[] features(double... values) {
        Feature[] features = new Feature[values.length];
        for (int i = 0; i < values.length; i++) {
            features[i] = new FeatureNode(i + 1, values[i]);
        }
        return features;
    }

}