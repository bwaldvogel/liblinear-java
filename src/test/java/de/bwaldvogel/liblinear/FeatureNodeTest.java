package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nl.jqno.equalsverifier.EqualsVerifier;
import nl.jqno.equalsverifier.Warning;


class FeatureNodeTest {

    @Test
    void testConstructorIndexZero() {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> new FeatureNode(0, 0))
            .withMessage("index must be > 0");
    }

    @Test
    void testConstructorIndexNegative() {
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> new FeatureNode(-1, 0))
            .withMessage("index must be > 0");
    }

    @Test
    void testConstructorHappy() {
        Feature fn = new FeatureNode(25, 27.39);
        assertThat(fn.getIndex()).isEqualTo(25);
        assertThat(fn.getValue()).isEqualTo(27.39);

        fn = new FeatureNode(1, -0.22222);
        assertThat(fn.getIndex()).isEqualTo(1);
        assertThat(fn.getValue()).isEqualTo(-0.22222);
    }

    @Test
    void testEqualsAndHashCodeContract() throws Exception {
        EqualsVerifier.forClass(FeatureNode.class)
            .usingGetClass()
            .suppress(Warning.NONFINAL_FIELDS)
            .verify();
    }

    @Test
    void testEqualsAndHashCodeNaNValue() throws Exception {
        FeatureNode a = new FeatureNode(1, Double.NaN);
        FeatureNode b = new FeatureNode(1, Double.NaN);
        assertThat(a).isEqualTo(b);
        assertThat(a).hasSameHashCodeAs(b);
    }

    @Test
    void testEqualsWithPositiveAndNegativeZeroValue() throws Exception {
        FeatureNode a = new FeatureNode(1, -0.0);
        FeatureNode b = new FeatureNode(1, +0.0);
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void testToString() throws Exception {
        assertThat(new FeatureNode(1, 2.0)).hasToString("FeatureNode(idx=1, value=2.0)");
    }

}
