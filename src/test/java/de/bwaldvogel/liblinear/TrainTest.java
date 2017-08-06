package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.TestUtils.writeToFile;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;


public class TrainTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Before
    public void reset() throws Exception {
        Linear.resetRandom();
        Linear.disableDebugOutput();
    }

    @Test
    public void testDoCrossValidationOnIrisDataSet() throws Exception {
        for (SolverType solver : SolverType.values()) {
            Train.main(new String[] {"-v", "5", "-s", "" + solver.getId(), "src/test/resources/iris.scale"});
        }
    }

    @Test
    public void testFindBestCOnIrisDataSet() throws Exception {
        Train.main(new String[] {"-C", "src/test/resources/iris.scale"});
    }

    @Test
    public void testParseCommandLine() {
        Train train = new Train();

        for (SolverType solver : SolverType.values()) {
            train.parse_command_line(new String[] {"-B", "5.3", "-s", "" + solver.getId(), "-p", "0.01", "model-filename"});
            assertThat(train.isFindC()).isFalse();
            assertThat(train.getNumFolds()).isEqualTo(0);
            Parameter param = train.getParameter();
            assertThat(param.solverType).isEqualTo(solver);
            // check default eps
            if (solver.getId() == 0 || solver.getId() == 2 //
                || solver.getId() == 5 || solver.getId() == 6) {
                assertThat(param.eps).isEqualTo(0.01);
            } else if (solver.getId() == 7) {
                assertThat(param.eps).isEqualTo(0.1);
            } else if (solver.getId() == 11) {
                assertThat(param.eps).isEqualTo(0.001);
            } else {
                assertThat(param.eps).isEqualTo(0.1);
            }
            // check if bias is set
            assertThat(train.getBias()).isEqualTo(5.3);
            assertThat(param.p).isEqualTo(0.01);
        }
    }

    @Test
    public void testParseCommandLine_FindC_NoSolverSpecified() {
        Train train = new Train();

        train.parse_command_line(new String[] {"-C", "model-filename"});
        assertThat(train.isFindC()).isTrue();
        assertThat(train.getNumFolds()).isEqualTo(5);
        Parameter param = train.getParameter();
        assertThat(param.solverType).isEqualTo(SolverType.L2R_L2LOSS_SVC);
        // check default eps
        assertThat(param.eps).isEqualTo(0.01);
        assertThat(param.p).isEqualTo(0.1);
    }

    @Test
    public void testParseCommandLine_FindC_SolverAndNumFoldsSpecified() {
        Train train = new Train();

        train.parse_command_line(new String[] {"-s", "0", "-v", "10", "-C", "model-filename"});
        assertThat(train.isFindC()).isTrue();
        assertThat(train.getNumFolds()).isEqualTo(10);
        Parameter param = train.getParameter();
        assertThat(param.solverType).isEqualTo(SolverType.L2R_LR);
        assertThat(param.eps).isEqualTo(0.01);
        assertThat(param.p).isEqualTo(0.1);
    }

    @Test
    // https://github.com/bwaldvogel/liblinear-java/issues/4
    public void testParseWeights() throws Exception {
        Train train = new Train();
        train.parse_command_line(new String[] {"-v", "10", "-c", "10", "-w1", "1.234", "model-filename"});
        Parameter parameter = train.getParameter();
        assertThat(parameter.weightLabel).isEqualTo(new int[] {1});
        assertThat(parameter.weight).isEqualTo(new double[] {1.234});

        train.parse_command_line(new String[] {"-w1", "1.234", "-w2", "0.12", "-w3", "7", "model-filename"});
        parameter = train.getParameter();
        assertThat(parameter.weightLabel).isEqualTo(new int[] {1, 2, 3});
        assertThat(parameter.weight).isEqualTo(new double[] {1.234, 0.12, 7});
    }

    @Test
    public void testReadProblem() throws Exception {

        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList(
                "1 1:1  3:1  4:1   6:1",
                "2 2:1  3:1  5:1   7:1",
                "1 3:1  5:1",
                "1 1:1  4:1  7:1",
                "2 4:1  5:1  7:1");

        writeToFile(file, lines);

        Train train = new Train();
        train.readProblem(file.getAbsolutePath());

        Problem prob = train.getProblem();
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new double[] { 1, 2, 1, 1, 2 });
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x).hasSize(prob.y.length);

        validate(prob);
    }

    @Test
    public void testReadProblemFromStream() throws Exception {
        String data = "1 1:1  3:1  4:1   6:1\n"
            + "2 2:1  3:1  5:1   7:1\n"
            + "1 3:1  5:1\n"
            + "1 1:1  4:1  7:1\n"
            + "2 4:1  5:1  7:1\n";

        Charset charset = Charset.forName("UTF-8");
        Problem prob = Train.readProblem(new ByteArrayInputStream(data.getBytes(charset)), charset, 1);
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(5);
        assertThat(prob.y).isEqualTo(new double[] {1, 2, 1, 1, 2});
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x).hasSize(prob.y.length);

        validate(prob);
    }

    /**
     * unit-test for Issue #1 (http://github.com/bwaldvogel/liblinear-java/issues#issue/1)
     */
    @Test
    public void testReadProblemEmptyLine() throws Exception {

        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList(
                "1 1:1  3:1  4:1   6:1",
                "2 ");

        writeToFile(file, lines);

        Problem prob = Train.readProblem(file, -1.0);
        assertThat(prob.bias).isEqualTo(-1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new double[] {1, 2});
        assertThat(prob.n).isEqualTo(6);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x).hasSize(prob.y.length);

        assertThat(prob.x[0]).hasSize(4);
        assertThat(prob.x[1]).hasSize(0);
    }

    @Test
    public void testReadUnsortedProblem() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList(
                "1 1:1  3:1  4:1   6:1",
                "2 2:1  3:1  5:1   7:1",
                "1 3:1  5:1  4:1"); // here's the mistake: not correctly sorted

        writeToFile(file, lines);

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
            fail("InvalidInputDataException expected");
        } catch (InvalidInputDataException e) {
            assertThat(e).hasMessage("indices must be sorted in ascending order");
        }
    }

    @Test
    public void testReadProblemWithInvalidIndex() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList(
                "1 1:1  3:1  4:1   6:1",
                "2 2:1  3:1  5:1  -4:1");

        writeToFile(file, lines);

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
            fail("InvalidInputDataException expected");
        } catch (InvalidInputDataException e) {
            assertThat(e).hasMessage("invalid index: -4");
        }
    }

    @Test
    public void testReadProblemWithZeroIndex() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Collections.singletonList("1 0:1  1:1");

        writeToFile(file, lines);

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
            fail("InvalidInputDataException expected");
        } catch (InvalidInputDataException e) {
            assertThat(e).hasMessage("invalid index: 0");
        }
    }

    @Test
    public void testReadWrongProblem() throws Exception {
        File file = temporaryFolder.newFile();

        List<String> lines = Arrays.asList(
                "1 1:1  3:1  4:1   6:1",
                "2 2:1  3:1  5:1   7:1",
                "1 3:1  5:a"); // here's the mistake: incomplete line

        writeToFile(file, lines);

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
            fail("InvalidInputDataException expected");
        } catch (InvalidInputDataException e) {
            assertThat(e).hasMessage("invalid value: a");
        }
    }

    private void validate(Problem prob) {
        for (Feature[] nodes : prob.x) {
            assertThat(nodes.length).isLessThanOrEqualTo(prob.n);
            for (Feature node : nodes) {
                // bias term
                if (prob.bias >= 0 && nodes[nodes.length - 1] == node) {
                    assertThat(node.getIndex()).isEqualTo(prob.n);
                    assertThat(node.getValue()).isEqualTo(prob.bias);
                } else {
                    assertThat(node.getIndex()).isLessThan(prob.n);
                }
            }
        }
    }

}
