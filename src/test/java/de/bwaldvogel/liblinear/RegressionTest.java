package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.assertj.core.data.Offset;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@RunWith(Parameterized.class)
public class RegressionTest {

    private static final Logger log = LoggerFactory.getLogger(RegressionTest.class);

    @Parameters(name = "{0}")
    public static Collection<TestParams> data() {
        List<TestParams> params = new ArrayList<>();
        for (String dataset : new String[] {"splice", "dna.scale"}) {
            for (SolverType solverType : SolverType.values()) {
                params.add(new TestParams(dataset, solverType));
            }
        }
        return params;
    }

    private final TestParams params;


    private static class TestParams {

        private final String     dataset;
        private final SolverType solverType;

        private TestParams(String dataset, SolverType solverType) {
            this.dataset = dataset;
            this.solverType = solverType;
        }

        @Override
        public String toString() {
            return "dataset: " + dataset + ", solver: " + solverType;
        }
    }

    public RegressionTest(TestParams params) {
        this.params = params;
    }

    @Test
    public void regressionTest() throws Exception {
        runRegressionTest(params.dataset, params.solverType);
    }

    private void runRegressionTest(String dataset, SolverType solverType) throws Exception {
        Linear.resetRandom();
        log.info("Running regression test for '{}'", params);
        File trainingFile = Paths.get("src/test/datasets", dataset, dataset).toFile();
        Problem problem = Train.readProblem(trainingFile, -1);
        Model model = Linear.train(problem, new Parameter(solverType, 1, 0.1));
        File testFile = Paths.get("src/test/datasets", dataset, dataset + ".t").toFile();
        Problem testProblem = Train.readProblem(testFile, -1);

        Path expectedFile = Paths.get("src/test/resources/regression", dataset, "predictions_" + solverType.name());
        List<String> expectedPredictions = Files.readAllLines(expectedFile, StandardCharsets.UTF_8);

        assertThat(expectedPredictions).hasSize(testProblem.l);
        assertThat(testProblem.x).hasSameSizeAs(expectedPredictions);

        for (int i = 0; i < testProblem.l; i++) {
            Feature[] x = testProblem.x[i];
            double[] predictedValues = new double[model.getNrClass()];
            if (solverType.isLogisticRegressionSolver()) {
                Linear.predictProbability(model, x, predictedValues);
            } else {
                Linear.predictValues(model, x, predictedValues);
            }

            List<Double> expectedValues = parseExpectedValues(expectedPredictions, i);

            Offset<Double> allowedOffset = Offset.offset(1e-9);
            if (model.getNrClass() == 2) {
                assertThat(expectedValues).hasSize(1);
                assertThat(predictedValues[0]).isEqualTo(expectedValues.get(0), allowedOffset);
            } else {
                assertThat(expectedValues).hasSameSizeAs(predictedValues);
                for (int n = 0; n < predictedValues.length; n++) {
                    assertThat(predictedValues[n]).isEqualTo(expectedValues.get(n), allowedOffset);
                }
            }
        }
    }

    private List<Double> parseExpectedValues(List<String> expectedPredictions, int i) {
        return Stream.of(expectedPredictions.get(i)
                .split(" "))
                .map(Double::parseDouble)
                .collect(Collectors.toList());
    }

}
