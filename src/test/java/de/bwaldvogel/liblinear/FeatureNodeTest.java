package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

import org.junit.Test;

public class FeatureNodeTest {

    @Test
    public void testConstructorIndexZero() {
        // since 1.5 there's no more exception here
        new FeatureNode(0, 0);
    }

    @Test
    public void testConstructorIndexNegative() {
        try {
            new FeatureNode(-1, 0);
            fail("IllegalArgumentException");
        } catch (Exception e) {
            assertThat(e).hasMessage("index must be >= 0");
        }
    }

    @Test
    public void testConstructorHappy() {
        Feature fn = new FeatureNode(25, 27.39);
        assertThat(fn.getIndex()).isEqualTo(25);
        assertThat(fn.getValue()).isEqualTo(27.39);

        fn = new FeatureNode(1, -0.22222);
        assertThat(fn.getIndex()).isEqualTo(1);
        assertThat(fn.getValue()).isEqualTo(-0.22222);
    }
}
