package de.bwaldvogel.liblinear;

import static org.junit.Assert.*;

import org.junit.Test;


public class SolverTypeTest {

    @Test
    public void testIsSupportVectorRegression() throws Exception {
        for (SolverType type : SolverType.values()) {
            boolean regressionSolver = type.isSupportVectorRegression();
            // from check_regression_model() in linear.cpp
            if (type == SolverType.L2R_L2LOSS_SVR || type == SolverType.L2R_L1LOSS_SVR_DUAL || type == SolverType.L2R_L2LOSS_SVR_DUAL) {
                assertTrue(type + " is a regression solver", regressionSolver);
            } else {
                assertFalse(type + " is not a regression solver", regressionSolver);
            }
        }
    }
}
