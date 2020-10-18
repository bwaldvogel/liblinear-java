package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.*;
import static org.assertj.core.api.Assertions.*;

import org.junit.jupiter.api.Test;


class SolverTypeTest {

    @Test
    void testIsSupportVectorRegression() throws Exception {
        for (SolverType type : SolverType.values()) {
            boolean regressionSolver = type.isSupportVectorRegression();
            // from check_regression_model() in linear.cpp
            if (type == L2R_L2LOSS_SVR || type == L2R_L1LOSS_SVR_DUAL || type == L2R_L2LOSS_SVR_DUAL) {
                assertThat(regressionSolver).withFailMessage(type + " is a regression solver").isTrue();
            } else {
                assertThat(regressionSolver).withFailMessage(type + " is not a regression solver").isFalse();
            }
        }
    }
}
