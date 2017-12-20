package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.TestUtils.repeat;
import static de.bwaldvogel.liblinear.TestUtils.writeToFile;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.powermock.api.mockito.PowerMockito;


public class LinearTest {

    private static Random random = new Random(12345);
    private static final int[] THREAD_COUNTS = new int[]{1, 2, 4, 8};

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Before
    public void reset() throws Exception {
        Linear.resetRandom();
        Linear.disableDebugOutput();
    }

    public static Model createRandomModel() {
        Model model = new Model();
        model.solverType = SolverType.L2R_LR;
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
        return model;
    }

    public static Problem createRandomProblem(int numClasses) {
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
    public void testTrainPredict() {
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
            for (int threadCount : solver == SolverType.L2R_LR ? THREAD_COUNTS : new int[]{1}) {
                for (double C = 0.1; C <= 100.; C *= 1.2) {

                    // compared the behavior with the C version
                    if (C < 0.2) if (solver == SolverType.L1R_L2LOSS_SVC) continue;
                    if (C < 0.7) if (solver == SolverType.L1R_LR) continue;

                    if (solver.isSupportVectorRegression()) {
                        continue;
                    }

                    Parameter param = new Parameter(solver, C, 0.1, 0.1);
                    param.setThreadCount(threadCount);
                    Model model = Linear.train(prob, param);

                    double[] featureWeights = model.getFeatureWeights();
                    if (solver == SolverType.MCSVM_CS) {
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
                            assertThat(estimates[(int) probabilityPrediction]).isGreaterThanOrEqualTo(1.0 / model.getNrClass());
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
    }

    @Test
    public void testCrossValidation() throws Exception {
        int numClasses = random.nextInt(10) + 1;

        Problem prob = createRandomProblem(numClasses);

        Parameter param = new Parameter(SolverType.L2R_LR, 10, 0.01);
        int nr_fold = 10;
        double[] target = new double[prob.l];
        Linear.crossValidation(prob, param, nr_fold, target);

        for (double clazz : target) {
            assertThat(clazz).isGreaterThanOrEqualTo(0).isLessThan(numClasses);
        }
    }

    @Test
    public void testLoadSaveModel() throws Exception {

        for (SolverType solverType : SolverType.values()) {
            Model model = createRandomModel();
            model.solverType = solverType;

            File tempFile = temporaryFolder.newFile("modeltest-" + solverType);
            Linear.saveModel(tempFile, model);

            Model loadedModel = Linear.loadModel(tempFile);
            assertThat(loadedModel).isEqualTo(model);
        }
    }

    @Test
    public void testLoadEmptyModel() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList("solver_type L2R_LR",
                "nr_class 2",
                "label 1 2",
                "nr_feature 0",
                "bias -1.0",
                "w");
        writeToFile(file, lines);

        Model model = Model.load(file);
        assertThat(model.getSolverType()).isEqualTo(SolverType.L2R_LR);
        assertThat(model.getLabels()).containsExactly(1, 2);
        assertThat(model.getNrClass()).isEqualTo(2);
        assertThat(model.getNrFeature()).isEqualTo(0);
        assertThat(model.getFeatureWeights()).isEmpty();
        assertThat(model.getBias()).isEqualTo(-1.0);
    }

    @Test
    public void testLoadSimpleModel() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList("solver_type L2R_L2LOSS_SVR",
                "nr_class 2",
                "label 1 2",
                "nr_feature 6",
                "bias -1.0",
                "w",
                "0.1 0.2 0.3 ",
                "0.4 0.5 0.6 ");
        writeToFile(file, lines);

        Model model = Model.load(file);
        assertThat(model.getSolverType()).isEqualTo(SolverType.L2R_L2LOSS_SVR);
        assertThat(model.getLabels()).containsExactly(1, 2);
        assertThat(model.getNrClass()).isEqualTo(2);
        assertThat(model.getNrFeature()).isEqualTo(6);
        assertThat(model.getFeatureWeights()).containsExactly(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        assertThat(model.getBias()).isEqualTo(-1.0, offset(0.001));
    }

    @Test
    public void testLoadIllegalModel() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList("solver_type L2R_L2LOSS_SVR",
                "nr_class 2",
                "label 1 2",
                "nr_feature 10",
                "bias -1.0",
                "w",
                "0.1 0.2 0.3 ",
                "0.4 0.5 " + repeat("0", 1024));
        writeToFile(file, lines);

        try {
            Model.load(file);
            fail("RuntimeException expected");
        } catch (RuntimeException e) {
            String x = repeat("0", 128);
            assertThat(e).hasMessage("illegal weight in model file at index 5, with string content '" + x
                    + "', is not terminated with a whitespace character, or is longer than expected (128 characters max).");
        }
    }

    @Test
    public void testTrainUnsortedProblem() {
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

        Parameter param = new Parameter(SolverType.L2R_LR, 10, 0.1);
        try {
            Linear.train(prob, param);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).contains("nodes").contains("sorted").contains("ascending").contains("order");
        }
    }

    @Test
    public void testTrainTooLargeProblem() {
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
            if (solverType.isSupportVectorRegression()) continue;
            Parameter param = new Parameter(solverType, 10, 0.1);
            try {
                Linear.train(prob, param);
                fail("IllegalArgumentException expected");
            } catch (IllegalArgumentException e) {
                assertThat(e.getMessage()).contains("number of classes").contains("too large");
            }
        }
    }

    @Test
    public void testPredictProbabilityWrongSolver() throws Exception {
        Problem prob = new Problem();
        prob.l = 1;
        prob.n = 1;
        prob.x = new FeatureNode[prob.l][];
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = new FeatureNode[] {};
            prob.y[i] = i;
        }

        SolverType solverType = SolverType.L2R_L1LOSS_SVC_DUAL;
        Parameter param = new Parameter(solverType, 10, 0.1);
        Model model = Linear.train(prob, param);
        try {
            Linear.predictProbability(model, prob.x[0], new double[1]);
            fail("IllegalArgumentException expected");
        } catch (IllegalArgumentException e) {
            assertThat(e.getMessage()).isEqualTo("probability output is only supported for logistic regression." //
                + " This is currently only supported by the following solvers:" //
                + " L2R_LR, L1R_LR, L2R_LR_DUAL");
        }
    }

    @Test
    public void testRealloc() {

        int[] f = new int[] {1, 2, 3};
        f = Linear.copyOf(f, 5);
        f[3] = 4;
        f[4] = 5;
        assertThat(f).isEqualTo(new int[] {1, 2, 3, 4, 5});
    }

    @Test
    public void testAtoi() {
        assertThat(Linear.atoi("+25")).isEqualTo(25);
        assertThat(Linear.atoi("-345345")).isEqualTo(-345345);
        assertThat(Linear.atoi("+0")).isEqualTo(0);
        assertThat(Linear.atoi("0")).isEqualTo(0);
        assertThat(Linear.atoi("2147483647")).isEqualTo(Integer.MAX_VALUE);
        assertThat(Linear.atoi("-2147483648")).isEqualTo(Integer.MIN_VALUE);
    }

    @Test(expected = NumberFormatException.class)
    public void testAtoiInvalidData() {
        Linear.atoi("+");
    }

    @Test(expected = NumberFormatException.class)
    public void testAtoiInvalidData2() {
        Linear.atoi("abc");
    }

    @Test(expected = NumberFormatException.class)
    public void testAtoiInvalidData3() {
        Linear.atoi(" ");
    }

    @Test
    public void testAtof() {
        assertThat(Linear.atof("+25")).isEqualTo(25);
        assertThat(Linear.atof("-25.12345678")).isEqualTo(-25.12345678);
        assertThat(Linear.atof("0.345345299")).isEqualTo(0.345345299);
    }

    @Test(expected = NumberFormatException.class)
    public void testAtofInvalidData() {
        Linear.atof("0.5t");
    }

    @Test
    public void testSaveModelWithIOException() throws Exception {
        Model model = createRandomModel();

        Writer out = PowerMockito.mock(Writer.class);

        IOException ioException = new IOException("some reason");

        doThrow(ioException).when(out).flush();

        try {
            Linear.saveModel(out, model);
            fail("IOException expected");
        } catch (IOException e) {
            assertThat(e).isEqualTo(ioException);
        }

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
    public void testTranspose() throws Exception {
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
    public void testTranspose2() throws Exception {
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
    public void testTranspose3() throws Exception {

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
        assertThat(transposed.x).hasSize(4);
        assertThat(transposed.x[0]).hasSize(2);
        assertThat(transposed.x[1]).hasSize(2);
        assertThat(transposed.x[2]).hasSize(2);
        assertThat(transposed.x[3]).hasSize(2);

        assertThat(transposed.x[0][0]).isEqualTo(new FeatureNode(1, 2));
        assertThat(transposed.x[0][1]).isEqualTo(new FeatureNode(2, 9));

        assertThat(transposed.x[1][0]).isEqualTo(new FeatureNode(2, 7));
        assertThat(transposed.x[1][1]).isEqualTo(new FeatureNode(3, 1));

        assertThat(transposed.x[2][0]).isEqualTo(new FeatureNode(1, 1));
        assertThat(transposed.x[2][1]).isEqualTo(new FeatureNode(2, 3));

        assertThat(transposed.x[3][0]).isEqualTo(new FeatureNode(1, 3));
        assertThat(transposed.x[3][1]).isEqualTo(new FeatureNode(2, 3));
    }

    private static double[] randomWeights(int count) {
        double[] weights = new double[count];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble();
        }
        return weights;
    }

    private static Problem createBinaryProblem(int exampleCount, double[] weights) {
        Problem prob = new Problem();
        prob.bias = -1;
        prob.l = exampleCount;
        prob.n = weights.length + 1;
        prob.x = new FeatureNode[prob.l][];
        prob.y = new double[prob.l];

        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = new FeatureNode[weights.length + 1];
            double dotProduct = 0;
            for (int j = 0; j < weights.length; j++) {
                double nextValue = random.nextDouble();
                dotProduct += weights[j] * nextValue;
                prob.x[i][j] = new FeatureNode(j + 1, nextValue);
            }
            prob.x[i][weights.length] = new FeatureNode(weights.length + 1, 1);

            // ~50% of examples will have a dot product high enough to garner a positive label:
            prob.y[i] = dotProduct >= (weights.length * 0.25) ? 1 : -1;
        }
        return prob;
    }

    /**
     * Creates a perfectly separable binary problem and tests the accuracy of the resultant model
     */
    @Test
    public void testBinaryLR() {
        double[] weights = randomWeights(5);
        Problem trainingData = createBinaryProblem(3000, weights);
        Problem evaluationData = createBinaryProblem(1000, weights);

        for (SolverType solver : SolverType.values()) {
            for (int threadCount : solver == SolverType.L2R_LR ? THREAD_COUNTS : new int[]{1}) {
                for (double C = 0.1; C <= 100.; C *= 1.2) {
                    if (!solver.isLogisticRegressionSolver()) continue;
                    // compared the behavior with the C version
                    if (C < 0.7) if (solver == SolverType.L1R_LR) continue;

                    Parameter param = new Parameter(solver, C, 0.1, 0.1);
                    param.setThreadCount(threadCount);
                    Model model = Linear.train(trainingData, param);

                    int mistakes = 0;
                    for (int i = 0; i < evaluationData.x.length; i++) {
                        double prediction = Linear.predict(model, evaluationData.x[i]);
                        if (prediction != evaluationData.y[i]) mistakes++;
                    }

                    assertThat(mistakes).as("mistakes with solver " + solver)
                        .isLessThan((int) (evaluationData.x.length * 0.1));
                }
            }
        }
    }
}
