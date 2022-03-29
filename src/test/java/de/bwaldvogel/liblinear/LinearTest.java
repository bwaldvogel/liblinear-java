package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.*;
import static de.bwaldvogel.liblinear.TestUtils.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;


class LinearTest {

    private static final Random random = new Random(12345);

    @BeforeEach
    public void reset() throws Exception {
        Linear.disableDebugOutput();
    }

    static Model createRandomModel() {
        return createRandomModel(L2R_LR);
    }

    static Model createRandomModel(SolverType solverType) {
        Model model = new Model();
        model.solverType = solverType;
        model.bias = 2;
        model.label = new int[] {1, Integer.MAX_VALUE, 2};
        model.w = new double[model.label.length * 300];
        for (int i = 0; i < model.w.length; i++) {
            // precision should be at least 1e-4
            model.w[i] = Math.round(random.nextDouble() * 100000.0) / 10000.0;
        }

        // force at least one value to be zero
        model.w[random.nextInt(model.w.length)] = 0.0;
        model.w[random.nextInt(model.w.length)] = -0.0;

        model.nr_feature = model.w.length / model.label.length - 1;
        model.nr_class = model.label.length;
        if (solverType.isOneClass()) {
            model.rho = random.nextDouble();
        }
        return model;
    }

    private static Problem createRandomProblem(int numClasses) {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = random.nextInt(100) + 1;
        prob.n = random.nextInt(100) + 1;
        prob.x = new FeatureNode[prob.l][];
        prob.y = new double[prob.l];

        for (int i = 0; i < prob.l; i++) {

            prob.y[i] = random.nextInt(numClasses);

            Set<Integer> randomNumbers = new TreeSet<>();
            int num = random.nextInt(prob.n) + 1;
            for (int j = 0; j < num; j++) {
                randomNumbers.add(random.nextInt(prob.n) + 1);
            }
            List<Integer> randomIndices = new ArrayList<>(randomNumbers);
            Collections.sort(randomIndices);

            prob.x[i] = new FeatureNode[randomIndices.size()];
            for (int j = 0; j < randomIndices.size(); j++) {
                prob.x[i][j] = new FeatureNode(randomIndices.get(j), random.nextDouble());
            }
        }
        return prob;
    }

    /**
     * create a very simple problem and check if the clearly separated examples are recognized as such
     */
    @Test
    void testTrainPredict() {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = 4;
        prob.n = 4;
        prob.x = new FeatureNode[4][];
        prob.x[0] = new FeatureNode[2];
        prob.x[1] = new FeatureNode[1];
        prob.x[2] = new FeatureNode[1];
        prob.x[3] = new FeatureNode[3];

        prob.x[0][0] = new FeatureNode(1, 1);
        prob.x[0][1] = new FeatureNode(2, 1);

        prob.x[1][0] = new FeatureNode(3, 1);
        prob.x[2][0] = new FeatureNode(3, 1);

        prob.x[3][0] = new FeatureNode(1, 2);
        prob.x[3][1] = new FeatureNode(2, 1);
        prob.x[3][2] = new FeatureNode(4, 1);

        prob.y = new double[4];
        prob.y[0] = 0;
        prob.y[1] = 1;
        prob.y[2] = 1;
        prob.y[3] = 0;

        for (SolverType solver : SolverType.values()) {
            if (solver.isOneClass()) {
                continue;
            }
            for (double C = 0.1; C <= 100.; C *= 1.2) {
                // compared the behavior with the C version
                if (C < 0.2)
                    if (solver == L1R_L2LOSS_SVC)
                        continue;
                if (C < 0.7)
                    if (solver == L1R_LR)
                        continue;

                if (solver.isSupportVectorRegression()) {
                    continue;
                }

                Parameter param = new Parameter(solver, C, 0.1, 0.1);
                Model model = Linear.train(prob, param);

                double[] featureWeights = model.getFeatureWeights();
                if (solver == MCSVM_CS) {
                    assertThat(featureWeights.length).isEqualTo(8);
                } else {
                    assertThat(featureWeights.length).isEqualTo(4);
                }

                int i = 0;
                for (double value : prob.y) {
                    double prediction = Linear.predict(model, prob.x[i]);
                    assertThat(prediction).as("prediction with solver " + solver).isEqualTo(value);
                    if (model.isProbabilityModel()) {
                        double[] estimates = new double[model.getNrClass()];
                        double probabilityPrediction = Linear.predictProbability(model, prob.x[i], estimates);
                        assertThat(probabilityPrediction).isEqualTo(prediction);
                        assertThat(estimates[(int)probabilityPrediction]).isGreaterThanOrEqualTo(1.0 / model.getNrClass());
                        double estimationSum = 0;
                        for (double estimate : estimates) {
                            estimationSum += estimate;
                        }
                        assertThat(estimationSum).isEqualTo(1.0, offset(0.001));
                    }
                    i++;
                }
            }
        }
    }

    @Test
    void testCrossValidation() throws Exception {
        int numClasses = random.nextInt(10) + 1;

        Problem prob = createRandomProblem(numClasses);

        Parameter param = new Parameter(L2R_LR, 10, 0.01);
        int nr_fold = 10;
        double[] target = new double[prob.l];
        Linear.crossValidation(prob, param, nr_fold, target);

        for (double clazz : target) {
            assertThat(clazz).isGreaterThanOrEqualTo(0).isLessThan(numClasses);
        }
    }

    @Test
    void testLoadSaveModel(@TempDir Path tempDir) throws Exception {
        for (SolverType solverType : SolverType.values()) {
            Model model = createRandomModel(solverType);

            Path tempFile = tempDir.resolve("modeltest-" + solverType);
            Linear.saveModel(tempFile, model);

            Model loadedModel = Linear.loadModel(tempFile);
            assertThat(loadedModel).isEqualTo(model);
        }
    }

    @Test
    void testLoadEmptyModel(@TempDir Path tempDir) throws Exception {
        Path modelPath = tempDir.resolve("empty-model");

        List<String> lines = Arrays.asList("solver_type L2R_LR",
            "nr_class 2",
            "label 1 2",
            "nr_feature 0",
            "bias -1.0",
            "w");
        writeToFile(modelPath, lines);

        Model model = Model.load(modelPath);
        assertThat(model.getSolverType()).isEqualTo(L2R_LR);
        assertThat(model.getLabels()).containsExactly(1, 2);
        assertThat(model.getNrClass()).isEqualTo(2);
        assertThat(model.getNrFeature()).isEqualTo(0);
        assertThat(model.getFeatureWeights()).isEmpty();
        assertThat(model.getBias()).isEqualTo(-1.0);
    }

    @Test
    void testLoadSimpleModel(@TempDir Path tempDir) throws Exception {
        Path modelPath = tempDir.resolve("simple-model");

        List<String> lines = Arrays.asList("solver_type L2R_L2LOSS_SVR",
            "nr_class 2",
            "label 1 2",
            "nr_feature 6",
            "bias -1.0",
            "w",
            "0.1 0.2 0.3 ",
            "0.4 0.5 0.6 ");
        writeToFile(modelPath, lines);

        Model model = Model.load(modelPath);
        assertThat(model.getSolverType()).isEqualTo(L2R_L2LOSS_SVR);
        assertThat(model.getLabels()).containsExactly(1, 2);
        assertThat(model.getNrClass()).isEqualTo(2);
        assertThat(model.getNrFeature()).isEqualTo(6);
        assertThat(model.getFeatureWeights()).containsExactly(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        assertThat(model.getBias()).isEqualTo(-1.0, offset(0.001));
    }

    @Test
    void testLoadIllegalModel(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("illegal-model");

        List<String> lines = Arrays.asList("solver_type L2R_L2LOSS_SVR",
            "nr_class 2",
            "label 1 2",
            "nr_feature 10",
            "bias -1.0",
            "w",
            "0.1 0.2 0.3 ",
            "0.4 0.5 " + repeat("0", 1024));
        writeToFile(file, lines);

        String x = repeat("0", 128);

        assertThatExceptionOfType(RuntimeException.class)
            .isThrownBy(() -> Model.load(file))
            .withMessage("illegal weight in model file at index 5, with string content '" + x
                + "', is not terminated with a whitespace character, or is longer than expected (128 characters max).");
    }

    @Test
    void testTrainUnsortedProblem() {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = 1;
        prob.n = 2;
        prob.x = new FeatureNode[4][];
        prob.x[0] = new FeatureNode[2];

        prob.x[0][0] = new FeatureNode(2, 1);
        prob.x[0][1] = new FeatureNode(1, 1);

        prob.y = new double[4];
        prob.y[0] = 0;

        Parameter param = new Parameter(L2R_LR, 10, 0.1);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.train(prob, param))
            .withMessageContainingAll("nodes", "sorted", "ascending", "order");
    }

    @Test
    void testTrainTooLargeProblem() {
        Problem prob = new Problem();
        prob.l = 1000;
        prob.n = 20000000;
        prob.x = new FeatureNode[prob.l][];
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = new FeatureNode[] {};
            prob.y[i] = i;
        }

        for (SolverType solverType : SolverType.values()) {
            if (solverType.isSupportVectorRegression() || solverType.isOneClass()) {
                continue;
            }
            Parameter param = new Parameter(solverType, 10, 0.1);

            assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> Linear.train(prob, param))
                .withMessageContainingAll("number of classes", "too large");
        }
    }

    @Test
    void testTrain_IllegalParameters_BiasWithOneClassSvm() {
        Problem prob = createRandomProblem(2);
        prob.bias = 1;

        Parameter param = new Parameter(ONECLASS_SVM, 10, 0.1);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.train(prob, param))
            .withMessage("prob->bias >=0, but this is ignored in ONECLASS_SVM");
    }

    @Test
    void testTrain_IllegalParameters_RegularizeBias() {
        Problem prob = createRandomProblem(2);
        prob.bias = -1;

        Parameter param = new Parameter(L2R_L1LOSS_SVR_DUAL, 10, 0.1);
        param.setRegularizeBias(false);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.train(prob, param))
            .withMessage("To not regularize bias, must specify -B 1 along with -R");

        prob.bias = 1;

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.train(prob, param))
            .withMessage("-R option supported only for solver L2R_LR, L2R_L2LOSS_SVC, L1R_L2LOSS_SVC, L1R_LR, and L2R_L2LOSS_SVR");

        param.setSolverType(L1R_LR);

        Model model = Linear.train(prob, param);
        assertThat(model.bias).isEqualTo(1.0);
    }

    @Test
    void testTrain_IllegalParameters_InitialSol() {
        Problem prob = createRandomProblem(2);

        Parameter param = new Parameter(L2R_L1LOSS_SVR_DUAL, 10, 0.1);
        param.setInitSol(new double[prob.n]);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.train(prob, param))
            .withMessage("Initial-solution specification supported only for solvers L2R_LR, L2R_L2LOSS_SVC, and L2R_L2LOSS_SVR");

        param.setSolverType(L2R_LR);

        Model model = Linear.train(prob, param);
        assertThat(model).isNotNull();
    }

    @Test
    void testPredictProbabilityWrongSolver() throws Exception {
        Problem prob = new Problem();
        prob.l = 1;
        prob.n = 1;
        prob.x = new FeatureNode[prob.l][];
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = new FeatureNode[] {};
            prob.y[i] = i;
        }

        Parameter param = new Parameter(L2R_L1LOSS_SVC_DUAL, 10, 0.1);
        Model model = Linear.train(prob, param);

        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> Linear.predictProbability(model, prob.x[0], new double[1]))
            .withMessage("probability output is only supported for logistic regression."
                + " This is currently only supported by the following solvers:"
                + " L2R_LR, L1R_LR, L2R_LR_DUAL");
    }

    @Test
    void testAtoi() {
        assertThat(Linear.atoi("+25")).isEqualTo(25);
        assertThat(Linear.atoi("-345345")).isEqualTo(-345345);
        assertThat(Linear.atoi("+0")).isEqualTo(0);
        assertThat(Linear.atoi("0")).isEqualTo(0);
        assertThat(Linear.atoi("2147483647")).isEqualTo(Integer.MAX_VALUE);
        assertThat(Linear.atoi("-2147483648")).isEqualTo(Integer.MIN_VALUE);
    }

    @Test
    void testAtoiInvalidData() {
        assertThatExceptionOfType(NumberFormatException.class)
            .isThrownBy(() -> Linear.atoi("+"));

        assertThatExceptionOfType(NumberFormatException.class)
            .isThrownBy(() -> Linear.atoi("abc"));

        assertThatExceptionOfType(NumberFormatException.class)
            .isThrownBy(() -> Linear.atoi(" "));
    }

    @Test
    void testAtof() {
        assertThat(Linear.atof("+25")).isEqualTo(25);
        assertThat(Linear.atof("-25.12345678")).isEqualTo(-25.12345678);
        assertThat(Linear.atof("0.345345299")).isEqualTo(0.345345299);
    }

    @Test
    void testAtofInvalidData() {
        assertThatExceptionOfType(NumberFormatException.class)
            .isThrownBy(() -> Linear.atof("0.5t"));
    }

    @Test
    void testSaveModelWithIOException() throws Exception {
        Model model = createRandomModel();

        Writer out = mock(Writer.class);

        IOException ioException = new IOException("some reason");

        doThrow(ioException).when(out).flush();

        assertThatExceptionOfType(IOException.class)
            .isThrownBy(() -> Linear.saveModel(out, model))
            .withMessage("some reason");

        verify(out).flush();
        verify(out, times(1)).close();
    }

    /**
     * compared input/output values with the C version (1.51)
     *
     * <pre>
     * IN:
     * res prob.l = 4
     * res prob.n = 4
     * 0: (2,1) (4,1)
     * 1: (1,1)
     * 2: (3,1)
     * 3: (2,2) (3,1) (4,1)
     *
     * TRANSPOSED:
     *
     * res prob.l = 4
     * res prob.n = 4
     * 0: (2,1)
     * 1: (1,1) (4,2)
     * 2: (3,1) (4,1)
     * 3: (1,1) (4,1)
     * </pre>
     */
    @Test
    void testTranspose() throws Exception {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = 4;
        prob.n = 4;
        prob.x = new FeatureNode[4][];
        prob.x[0] = new FeatureNode[2];
        prob.x[1] = new FeatureNode[1];
        prob.x[2] = new FeatureNode[1];
        prob.x[3] = new FeatureNode[3];

        prob.x[0][0] = new FeatureNode(2, 1);
        prob.x[0][1] = new FeatureNode(4, 1);

        prob.x[1][0] = new FeatureNode(1, 1);
        prob.x[2][0] = new FeatureNode(3, 1);

        prob.x[3][0] = new FeatureNode(2, 2);
        prob.x[3][1] = new FeatureNode(3, 1);
        prob.x[3][2] = new FeatureNode(4, 1);

        prob.y = new double[4];
        prob.y[0] = 0;
        prob.y[1] = 1;
        prob.y[2] = 1;
        prob.y[3] = 0;

        Problem transposed = Linear.transpose(prob);

        assertThat(transposed.x[0].length).isEqualTo(1);
        assertThat(transposed.x[1].length).isEqualTo(2);
        assertThat(transposed.x[2].length).isEqualTo(2);
        assertThat(transposed.x[3].length).isEqualTo(2);

        assertThat(transposed.x[0][0]).isEqualTo(new FeatureNode(2, 1));

        assertThat(transposed.x[1][0]).isEqualTo(new FeatureNode(1, 1));
        assertThat(transposed.x[1][1]).isEqualTo(new FeatureNode(4, 2));

        assertThat(transposed.x[2][0]).isEqualTo(new FeatureNode(3, 1));
        assertThat(transposed.x[2][1]).isEqualTo(new FeatureNode(4, 1));

        assertThat(transposed.x[3][0]).isEqualTo(new FeatureNode(1, 1));
        assertThat(transposed.x[3][1]).isEqualTo(new FeatureNode(4, 1));

        assertThat(transposed.y).isEqualTo(prob.y);
    }

    /**
     *
     * compared input/output values with the C version (1.51)
     *
     * <pre>
     * IN:
     * res prob.l = 5
     * res prob.n = 10
     * 0: (1,7) (3,3) (5,2)
     * 1: (2,1) (4,5) (5,3) (7,4) (8,2)
     * 2: (1,9) (3,1) (5,1) (10,7)
     * 3: (1,2) (2,2) (3,9) (4,7) (5,8) (6,1) (7,5) (8,4)
     * 4: (3,1) (10,3)
     *
     * TRANSPOSED:
     *
     * res prob.l = 5
     * res prob.n = 10
     * 0: (1,7) (3,9) (4,2)
     * 1: (2,1) (4,2)
     * 2: (1,3) (3,1) (4,9) (5,1)
     * 3: (2,5) (4,7)
     * 4: (1,2) (2,3) (3,1) (4,8)
     * 5: (4,1)
     * 6: (2,4) (4,5)
     * 7: (2,2) (4,4)
     * 8:
     * 9: (3,7) (5,3)
     * </pre>
     */
    @Test
    void testTranspose2() throws Exception {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = 5;
        prob.n = 10;
        prob.x = new FeatureNode[5][];
        prob.x[0] = new FeatureNode[3];
        prob.x[1] = new FeatureNode[5];
        prob.x[2] = new FeatureNode[4];
        prob.x[3] = new FeatureNode[8];
        prob.x[4] = new FeatureNode[2];

        prob.x[0][0] = new FeatureNode(1, 7);
        prob.x[0][1] = new FeatureNode(3, 3);
        prob.x[0][2] = new FeatureNode(5, 2);

        prob.x[1][0] = new FeatureNode(2, 1);
        prob.x[1][1] = new FeatureNode(4, 5);
        prob.x[1][2] = new FeatureNode(5, 3);
        prob.x[1][3] = new FeatureNode(7, 4);
        prob.x[1][4] = new FeatureNode(8, 2);

        prob.x[2][0] = new FeatureNode(1, 9);
        prob.x[2][1] = new FeatureNode(3, 1);
        prob.x[2][2] = new FeatureNode(5, 1);
        prob.x[2][3] = new FeatureNode(10, 7);

        prob.x[3][0] = new FeatureNode(1, 2);
        prob.x[3][1] = new FeatureNode(2, 2);
        prob.x[3][2] = new FeatureNode(3, 9);
        prob.x[3][3] = new FeatureNode(4, 7);
        prob.x[3][4] = new FeatureNode(5, 8);
        prob.x[3][5] = new FeatureNode(6, 1);
        prob.x[3][6] = new FeatureNode(7, 5);
        prob.x[3][7] = new FeatureNode(8, 4);

        prob.x[4][0] = new FeatureNode(3, 1);
        prob.x[4][1] = new FeatureNode(10, 3);

        prob.y = new double[5];
        prob.y[0] = 0;
        prob.y[1] = 1;
        prob.y[2] = 1;
        prob.y[3] = 0;
        prob.y[4] = 1;

        Problem transposed = Linear.transpose(prob);

        assertThat(transposed.x[0]).hasSize(3);
        assertThat(transposed.x[1]).hasSize(2);
        assertThat(transposed.x[2]).hasSize(4);
        assertThat(transposed.x[3]).hasSize(2);
        assertThat(transposed.x[4]).hasSize(4);
        assertThat(transposed.x[5]).hasSize(1);
        assertThat(transposed.x[7]).hasSize(2);
        assertThat(transposed.x[7]).hasSize(2);
        assertThat(transposed.x[8]).hasSize(0);
        assertThat(transposed.x[9]).hasSize(2);

        assertThat(transposed.x[0][0]).isEqualTo(new FeatureNode(1, 7));
        assertThat(transposed.x[0][1]).isEqualTo(new FeatureNode(3, 9));
        assertThat(transposed.x[0][2]).isEqualTo(new FeatureNode(4, 2));

        assertThat(transposed.x[1][0]).isEqualTo(new FeatureNode(2, 1));
        assertThat(transposed.x[1][1]).isEqualTo(new FeatureNode(4, 2));

        assertThat(transposed.x[2][0]).isEqualTo(new FeatureNode(1, 3));
        assertThat(transposed.x[2][1]).isEqualTo(new FeatureNode(3, 1));
        assertThat(transposed.x[2][2]).isEqualTo(new FeatureNode(4, 9));
        assertThat(transposed.x[2][3]).isEqualTo(new FeatureNode(5, 1));

        assertThat(transposed.x[3][0]).isEqualTo(new FeatureNode(2, 5));
        assertThat(transposed.x[3][1]).isEqualTo(new FeatureNode(4, 7));

        assertThat(transposed.x[4][0]).isEqualTo(new FeatureNode(1, 2));
        assertThat(transposed.x[4][1]).isEqualTo(new FeatureNode(2, 3));
        assertThat(transposed.x[4][2]).isEqualTo(new FeatureNode(3, 1));
        assertThat(transposed.x[4][3]).isEqualTo(new FeatureNode(4, 8));

        assertThat(transposed.x[5][0]).isEqualTo(new FeatureNode(4, 1));

        assertThat(transposed.x[6][0]).isEqualTo(new FeatureNode(2, 4));
        assertThat(transposed.x[6][1]).isEqualTo(new FeatureNode(4, 5));

        assertThat(transposed.x[7][0]).isEqualTo(new FeatureNode(2, 2));
        assertThat(transposed.x[7][1]).isEqualTo(new FeatureNode(4, 4));

        assertThat(transposed.x[9][0]).isEqualTo(new FeatureNode(3, 7));
        assertThat(transposed.x[9][1]).isEqualTo(new FeatureNode(5, 3));

        assertThat(transposed.y).isEqualTo(prob.y);
    }

    /**
     * compared input/output values with the C version (1.51)
     *
     * IN:
     * res prob.l = 3
     * res prob.n = 4
     * 0: (1,2) (3,1) (4,3)
     * 1: (1,9) (2,7) (3,3) (4,3)
     * 2: (2,1)
     *
     * TRANSPOSED:
     *
     * res prob.l = 3
     *      * res prob.n = 4
     * 0: (1,2) (2,9)
     * 1: (2,7) (3,1)
     * 2: (1,1) (2,3)
     * 3: (1,3) (2,3)
     *
     */
    @Test
    void testTranspose3() throws Exception {
        Problem prob = new Problem();
        prob.l = 3;
        prob.n = 4;
        prob.y = new double[3];
        prob.x = new FeatureNode[4][];
        prob.x[0] = new FeatureNode[3];
        prob.x[1] = new FeatureNode[4];
        prob.x[2] = new FeatureNode[1];
        prob.x[3] = new FeatureNode[1];

        prob.x[0][0] = new FeatureNode(1, 2);
        prob.x[0][1] = new FeatureNode(3, 1);
        prob.x[0][2] = new FeatureNode(4, 3);
        prob.x[1][0] = new FeatureNode(1, 9);
        prob.x[1][1] = new FeatureNode(2, 7);
        prob.x[1][2] = new FeatureNode(3, 3);
        prob.x[1][3] = new FeatureNode(4, 3);

        prob.x[2][0] = new FeatureNode(2, 1);

        prob.x[3][0] = new FeatureNode(3, 2);

        Problem transposed = Linear.transpose(prob);
        assertThat(transposed.x).hasDimensions(4, 2);

        assertThat(transposed.x[0][0]).isEqualTo(new FeatureNode(1, 2));
        assertThat(transposed.x[0][1]).isEqualTo(new FeatureNode(2, 9));

        assertThat(transposed.x[1][0]).isEqualTo(new FeatureNode(2, 7));
        assertThat(transposed.x[1][1]).isEqualTo(new FeatureNode(3, 1));

        assertThat(transposed.x[2][0]).isEqualTo(new FeatureNode(1, 1));
        assertThat(transposed.x[2][1]).isEqualTo(new FeatureNode(2, 3));

        assertThat(transposed.x[3][0]).isEqualTo(new FeatureNode(1, 3));
        assertThat(transposed.x[3][1]).isEqualTo(new FeatureNode(2, 3));
    }

    @Test
    void testFindBestParametersOnIrisDataSet() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/resources/iris.scale"), -1);
        Parameter param = new Parameter(L2R_L2LOSS_SVC, 1, 0.001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(4);
        assertThat(result.getBestScore()).isEqualTo(0.88);
        assertThat(result.getBestP()).isEqualTo(-1);
    }

    @Test
    void testFindBestParameterC_IllegalSolver() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/resources/iris.scale"), -1);

        EnumSet<SolverType> supportedSolvers = EnumSet.of(L2R_LR, L2R_L2LOSS_SVC, L2R_L2LOSS_SVR);
        for (SolverType illegalSolver : EnumSet.complementOf(supportedSolvers)) {
            Parameter param = new Parameter(illegalSolver, 1, 0.001, 0.1);
            assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> Linear.findParameters(problem, param, 5, -1, -1))
                .withMessage("Unsupported solver: " + illegalSolver);
        }
    }

    @Test
    void testFindBestParametersOnSpliceDataSet() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/datasets/splice/splice"), -1);
        Parameter param = new Parameter(L2R_L2LOSS_SVC, 1, 0.001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(0.001953125);
        assertThat(result.getBestScore()).isEqualTo(0.81);
        assertThat(result.getBestP()).isEqualTo(-1);
    }

    @Test
    void testFindBestParametersOnSpliceDataSet_L2R_LR() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/datasets/splice/splice"), -1);
        Parameter param = new Parameter(L2R_LR, 1, 0.001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(0.015625);
        assertThat(result.getBestScore()).isEqualTo(0.812);
        assertThat(result.getBestP()).isEqualTo(-1);
    }

    @Test
    void testFindBestParametersOnSpliceDataSet_L2R_L2LOSS_SVR() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/datasets/splice/splice"), -1);
        Parameter param = new Parameter(L2R_L2LOSS_SVR, 1, 0.001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(0.00390625);
        assertThat(result.getBestScore()).isEqualTo(0.5699399182191544, Offset.offset(0.0000001));
        assertThat(result.getBestP()).isEqualTo(0.0);
    }

    @Test
    void testFindBestParametersOnDnaScaleDataSet() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/datasets/dna.scale/dna.scale"), -1);
        Parameter param = new Parameter(L2R_L2LOSS_SVC, 1, 0.001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(0.015625);
        assertThat(result.getBestScore()).isEqualTo(0.9475);
        assertThat(result.getBestP()).isEqualTo(-1);
    }

    @Test
    void testFindBestParametersOnDnaScaleDataSet_L2R_L2LOSS_SVR() throws Exception {
        Problem problem = Train.readProblem(Paths.get("src/test/datasets/dna.scale/dna.scale"), -1);
        Parameter param = new Parameter(L2R_L2LOSS_SVR, 1, 0.0001, 0.1);
        ParameterSearchResult result = Linear.findParameters(problem, param, 5, -1, -1);
        assertThat(result.getBestC()).isEqualTo(0.015625);
        assertThat(result.getBestScore()).isEqualTo(0.29743037982927906, Offset.offset(0.0000001));
        assertThat(result.getBestP()).isEqualTo(0.15);
    }

    @Test
    void testGetVersion() throws Exception {
        assertThat(Linear.getVersion()).isEqualTo(244);
    }

}
