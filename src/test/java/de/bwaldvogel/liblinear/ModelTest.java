package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nl.jqno.equalsverifier.EqualsVerifier;
import nl.jqno.equalsverifier.Warning;


class ModelTest {

    @Test
    void testEqualsAndHashCodeContract() throws Exception {
        EqualsVerifier.forClass(Model.class)
            .suppress(Warning.NONFINAL_FIELDS)
            .verify();
    }

    @Test
    void testEqualsAndHashCode_EmptyModel() throws Exception {
        Model model1 = new Model();
        Model model2 = new Model();

        assertThat(model1).hasSameHashCodeAs(model2);
        assertThat(model1).isEqualTo(model2);
    }

    @Test
    void testEqualsAndHashCode_NegativeZeroesInWeightArray() throws Exception {
        Model model1 = new Model();
        Model model2 = new Model();

        model1.w = new double[] {1.0, 2.0, -0.0, -0.00};
        model2.w = new double[] {1.0, 2.0, +0.0, -0.000};

        assertThat(model1).hasSameHashCodeAs(model2);
        assertThat(model1).isEqualTo(model2);
    }

}
