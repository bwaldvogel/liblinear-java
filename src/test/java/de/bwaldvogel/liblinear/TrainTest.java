package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.*;
import static de.bwaldvogel.liblinear.TestUtils.*;
import static org.assertj.core.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;


class TrainTest {

    @BeforeEach
    public void reset() throws Exception {
        Linear.disableDebugOutput();
    }

    @Test
    void testDoCrossValidationOnIrisDataSet() throws Exception {
        for (SolverType solver : SolverType.values()) {
            if (solver.isOneClass()) {
                continue;
            }
            Train.main(new String[] {"-v", "5", "-s", "" + solver.getId(), "src/test/resources/iris.scale"});
        }
    }

    @Test
    void testFindBestCOnIrisDataSet() throws Exception {
        Train.main(new String[] {"-C", "src/test/resources/iris.scale"});
    }

    @Test
    void testFindBestCOnIrisDataSet_L2R_L2LOSS_SVR_DUAL() throws Exception {
        Train.main(new String[] {"-s", "11", "-C", "src/test/resources/iris.scale"});
    }

    @Test
    void testFindBestCOnSpliceDataSet_L2R_L2LOSS_SVR_DUAL() throws Exception {
        Train.main(new String[] {"-s", "11", "-C", "src/test/datasets/splice/splice"});
    }

    @Test
    void testParseCommandLine() {
        Train train = new Train();

        for (SolverType solver : SolverType.values()) {
            train.parse_command_line(new String[] {"-B", "5.3", "-s", "" + solver.getId(), "-p", "0.01", "model-filename"});
            assertThat(train.isFindParameters()).isFalse();
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
                assertThat(param.eps).isEqualTo(0.0001);}
            else if (solver.getId() == 21) {
                assertThat(param.eps).isEqualTo(0.01);
            } else {
                assertThat(param.eps).isEqualTo(0.1);
            }
            // check if bias is set
            assertThat(train.getBias()).isEqualTo(5.3);
            assertThat(param.p).isEqualTo(0.01);
        }
    }

    @Test
    void testParseCommandLine_FindC_NoSolverSpecified() {
        Train train = new Train();

        train.parse_command_line(new String[] {"-C", "model-filename"});
        assertThat(train.isFindParameters()).isTrue();
        assertThat(train.getNumFolds()).isEqualTo(5);
        Parameter param = train.getParameter();
        assertThat(param.solverType).isEqualTo(L2R_L2LOSS_SVC);
        // check default eps
        assertThat(param.eps).isEqualTo(0.01);
        assertThat(param.p).isEqualTo(0.1);
    }

    @Test
    void testParseCommandLine_FindC_SolverAndNumFoldsSpecified() {
        Train train = new Train();

        train.parse_command_line(new String[] {"-s", "0", "-v", "10", "-C", "model-filename"});
        assertThat(train.isFindParameters()).isTrue();
        assertThat(train.getNumFolds()).isEqualTo(10);
        Parameter param = train.getParameter();
        assertThat(param.solverType).isEqualTo(L2R_LR);
        assertThat(param.eps).isEqualTo(0.01);
        assertThat(param.p).isEqualTo(0.1);
    }

    @Test
    // https://github.com/bwaldvogel/liblinear-java/issues/4
    void testParseWeights() throws Exception {
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
    void testParseCommandLine_regularizeBias() throws Exception {
        Train train = new Train();
        train.parse_command_line(new String[] {"-R", "model-filename"});
        Parameter parameter = train.getParameter();
        assertThat(parameter.regularize_bias).isFalse();

        train.parse_command_line(new String[] {"model-filename"});
        parameter = train.getParameter();
        assertThat(parameter.regularize_bias).isTrue();
    }

    @Test
    void testReadProblem(@TempDir Path tempDir) throws Exception {
        Path problemPath = tempDir.resolve("problem");

        List<String> lines = Arrays.asList(
            "1 1:1  3:1  4:1   6:1",
            "2 2:1  3:1  5:1   7:1",
            "1 3:1  5:1",
            "1 1:1  4:1  7:1",
            "2 4:1  5:1  7:1");

        writeToFile(problemPath, lines);

        Train train = new Train();
        train.readProblem(problemPath);

        Problem prob = train.getProblem();
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new double[] {1, 2, 1, 1, 2});
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x.length).isEqualTo(prob.y.length);

        validate(prob);
    }

    @Test
    void testReadProblemFromStream() throws Exception {
        String data = "1 1:1  3:1  4:1   6:1\n"
            + "2 2:1  3:1  5:1   7:1\n"
            + "1 3:1  5:1\n"
            + "1 1:1  4:1  7:1\n"
            + "2 4:1  5:1  7:1\n";

        Charset charset = StandardCharsets.UTF_8;
        Problem prob = Train.readProblem(new ByteArrayInputStream(data.getBytes(charset)), charset, 1);
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(5);
        assertThat(prob.y).isEqualTo(new double[] {1, 2, 1, 1, 2});
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x.length).isEqualTo(prob.y.length);

        validate(prob);
    }

    /**
     * unit-test for Issue #1 (http://github.com/bwaldvogel/liblinear-java/issues#issue/1)
     */
    @Test
    void testReadProblemEmptyLine(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("problem");

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
        assertThat(prob.x.length).isEqualTo(prob.y.length);

        assertThat(prob.x[0]).hasSize(4);
        assertThat(prob.x[1]).hasSize(0);
    }

    @Test
    void testReadUnsortedProblem(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("problem");

        List<String> lines = Arrays.asList(
            "1 1:1  3:1  4:1   6:1",
            "2 2:1  3:1  5:1   7:1",
            "1 3:1  5:1  4:1"); // here's the mistake: not correctly sorted

        writeToFile(file, lines);

        Train train = new Train();

        assertThatExceptionOfType(InvalidInputDataException.class)
            .isThrownBy(() -> train.readProblem(file))
            .withMessage("indices must be sorted in ascending order");
    }

    @Test
    void testReadProblemWithInvalidIndex(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("problem");

        List<String> lines = Arrays.asList(
            "1 1:1  3:1  4:1   6:1",
            "2 2:1  3:1  5:1  -4:1");

        writeToFile(file, lines);

        Train train = new Train();

        assertThatExceptionOfType(InvalidInputDataException.class)
            .isThrownBy(() -> train.readProblem(file))
            .withMessage("invalid index: -4");
    }

    @Test
    void testReadProblemWithZeroIndex(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("problem");

        List<String> lines = Collections.singletonList("1 0:1  1:1");

        writeToFile(file, lines);

        Train train = new Train();

        assertThatExceptionOfType(InvalidInputDataException.class)
            .isThrownBy(() -> train.readProblem(file))
            .withMessage("invalid index: 0");
    }

    @Test
    void testReadWrongProblem(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("problem");

        List<String> lines = Arrays.asList(
            "1 1:1  3:1  4:1   6:1",
            "2 2:1  3:1  5:1   7:1",
            "1 3:1  5:a"); // here's the mistake: incomplete line

        writeToFile(file, lines);

        Train train = new Train();

        assertThatExceptionOfType(InvalidInputDataException.class)
            .isThrownBy(() -> train.readProblem(file))
            .withMessage("invalid value: a");
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
