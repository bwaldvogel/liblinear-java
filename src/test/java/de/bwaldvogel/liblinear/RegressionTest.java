package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


class RegressionTest {

    private static final Logger log = LoggerFactory.getLogger(RegressionTest.class);

    protected static final List<SolverType> SOLVERS = Stream.of(SolverType.values())
        .filter(solver -> !solver.isOneClass())
        .collect(Collectors.toList());

    private static Collection<TestParams> data() {
        List<TestParams> params = new ArrayList<>();
        for (String dataset : new String[] {"splice", "dna.scale"}) {
            for (SolverType solverType : SOLVERS) {
                params.add(new TestParams(dataset, solverType, getExpectedAccuracy(dataset, solverType)));
            }
        }
        return params;
    }

    private static Double getExpectedAccuracy(String dataset, SolverType solverType) {
        if (solverType.isSupportVectorRegression() || solverType.isOneClass()) {
            return null;
        }
        switch (dataset) {
            case "splice":
                switch (solverType) {
                    case L2R_LR:
                        return 0.8423;
                    case L2R_L2LOSS_SVC_DUAL:
                        return 0.8386;
                    case L2R_L2LOSS_SVC:
                        return 0.8432;
                    case L2R_L1LOSS_SVC_DUAL:
                        return 0.8382;
                    case MCSVM_CS:
                        return 0.8377;
                    case L1R_L2LOSS_SVC:
                        return 0.8478;
                    case L1R_LR:
                        return 0.8473;
                    case L2R_LR_DUAL:
                        return 0.8423;
                }
            case "dna.scale":
                switch (solverType) {
                    case L2R_LR:
                        return 0.9511;
                    case L2R_L2LOSS_SVC_DUAL:
                        return 0.9452;
                    case L2R_L2LOSS_SVC:
                        return 0.9469;
                    case L2R_L1LOSS_SVC_DUAL:
                        return 0.9477;
                    case MCSVM_CS:
                        return 0.9292;
                    case L1R_L2LOSS_SVC:
                        return 0.9553;
                    case L1R_LR:
                        return 0.9536;
                    case L2R_LR_DUAL:
                        return 0.9486;
                }
            default:
                throw new IllegalArgumentException("Unknown expectation: " + dataset + ", " + solverType);
        }
    }

    private static class TestParams {

        private final String     dataset;
        private final SolverType solverType;
        private final Double     expectedAccuracy;

        private TestParams(String dataset, SolverType solverType, Double expectedAccuracy) {
            this.dataset = dataset;
            this.solverType = solverType;
            this.expectedAccuracy = expectedAccuracy;
        }

        @Override
        public String toString() {
            return "dataset: " + dataset + ", solver: " + solverType;
        }
    }

    @ParameterizedTest
    @MethodSource("data")
    void regressionTest(TestParams params) throws Exception {
        log.info("Running regression test for '{}'", params);
        runRegressionTest(params.dataset, params.solverType, params.expectedAccuracy);
    }

    private void runRegressionTest(String dataset, SolverType solverType, Double expectedAccuracy) throws Exception {
        Linear.resetRandom();
        Path trainingFile = Paths.get("src/test/datasets", dataset, dataset);
        Problem problem = Train.readProblem(trainingFile, -1);
        Model model = Linear.train(problem, new Parameter(solverType, 1, 0.1));
        Path testFile = Paths.get("src/test/datasets", dataset, dataset + ".t");
        Problem testProblem = Train.readProblem(testFile, -1);

        Path expectedFile = Paths.get("src/test/resources/regression", dataset, "predictions_" + solverType.name());
        final List<String> expectedPredictions;
        if (!Files.exists(expectedFile)) {
            expectedPredictions = Collections.emptyList();
            log.warn("Recording predictions to {}", expectedFile);
        } else {
            expectedPredictions = Files.readAllLines(expectedFile, StandardCharsets.UTF_8);
            assertThat(expectedPredictions).hasSize(testProblem.l);
            assertThat(testProblem.x.length).isEqualTo(expectedPredictions.size());
        }

        int correctPredictions = 0;

        for (int i = 0; i < testProblem.l; i++) {
            Feature[] x = testProblem.x[i];
            double[] predictedValues = new double[model.getNrClass()];
            final double prediction;
            if (solverType.isLogisticRegressionSolver()) {
                prediction = Linear.predictProbability(model, x, predictedValues);
            } else {
                prediction = Linear.predictValues(model, x, predictedValues);
            }

            if (expectedAccuracy != null) {
                int expectation = (int)testProblem.y[i];
                int actual = (int)prediction;
                if (actual == expectation) {
                    correctPredictions++;
                }
            }

            if (expectedPredictions.isEmpty()) {
                final String line;
                if (model.getNrClass() == 2) {
                    line = predictedValues[0] + "\n";
                } else {
                    line = Arrays.stream(predictedValues)
                        .mapToObj(Double::toString)
                        .collect(Collectors.joining(" ")) + " \n";
                }
                Files.createDirectories(expectedFile.getParent());
                Files.write(expectedFile, line.getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
                continue;
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

        if (expectedAccuracy != null) {
            double accuracy = correctPredictions / (double)testProblem.l;
            assertThat(accuracy).isEqualTo(expectedAccuracy.doubleValue(), Offset.offset(1e-4));
        }
    }

    @Test
    void testOneClass(@TempDir Path tempDir) throws Exception {
        Linear.resetRandom();
        Path trainingFile = Paths.get("src/test/datasets/splice/splice");

        Path spliceClass1 = tempDir.resolve("splice-class-1");
        Path spliceClass2 = tempDir.resolve("splice-class-2");

        for (String line : Files.readAllLines(trainingFile, StandardCharsets.ISO_8859_1)) {
            final Path targetFile;
            if (line.startsWith("+1")) {
                targetFile = spliceClass1;
            } else {
                targetFile = spliceClass2;
            }
            Files.write(targetFile, Arrays.asList(line), StandardCharsets.UTF_8, StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        Problem problem1 = Train.readProblem(spliceClass1, StandardCharsets.UTF_8, -1);
        Parameter param = new Parameter(SolverType.ONECLASS_SVM, 1, 0.01);
        param.setNu(0.1);
        Model model = Linear.train(problem1, param);

        Model expectedModel = Model.load(Paths.get("src/test/resources/regression/splice/one_class_model"));
        assertThat(expectedModel).isEqualTo(model);

        Problem problem2 = Train.readProblem(spliceClass2, StandardCharsets.UTF_8, -1);

        // expected values determined with C-version of liblinear (v2.41)
        assertThat(calculatePredictionAccuracy(model, problem1)).isEqualTo(0.897485, Offset.strictOffset(1e-6));
        assertThat(calculatePredictionAccuracy(model, problem2)).isEqualTo(0.0703934, Offset.strictOffset(1e-6));
    }

    private static double calculatePredictionAccuracy(Model model, Problem problem) {
        int correct = 0;
        for (Feature[] x : problem.x) {
            double prediction = Linear.predict(model, x);
            if (prediction == problem.y[0]) {
                correct++;
            }
        }
        return (double)correct / problem.l;
    }

    private List<Double> parseExpectedValues(List<String> expectedPredictions, int i) {
        return Stream.of(expectedPredictions.get(i)
            .split(" "))
            .map(Double::parseDouble)
            .collect(Collectors.toList());
    }

}
