package de.bwaldvogel.liblinear;

import static org.fest.assertions.Assertions.assertThat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;


public class TrainTest {

    @Test
    public void testParseCommandLine() {
        Train train = new Train();

        for (SolverType solver : SolverType.values()) {
            train.parse_command_line(new String[] {"-B", "5.3", "-s", "" + solver.getId(), "-p", "0.01", "model-filename"});
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

        File file = File.createTempFile("svm", "test");
        file.deleteOnExit();

        Collection<String> lines = new ArrayList<String>();
        lines.add("1 1:1  3:1  4:1   6:1");
        lines.add("2 2:1  3:1  5:1   7:1");
        lines.add("1 3:1  5:1");
        lines.add("1 1:1  4:1  7:1");
        lines.add("2 4:1  5:1  7:1");
        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        try {
            for (String line : lines)
                writer.append(line).append("\n");
        }
        finally {
            writer.close();
        }

        Train train = new Train();
        train.readProblem(file.getAbsolutePath());

        Problem prob = train.getProblem();
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new double[] {1, 2, 1, 1, 2});
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x).hasSize(prob.y.length);

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

    /**
     * unit-test for Issue #1 (http://github.com/bwaldvogel/liblinear-java/issues#issue/1)
     */
    @Test
    public void testReadProblemEmptyLine() throws Exception {

        File file = File.createTempFile("svm", "test");
        file.deleteOnExit();

        Collection<String> lines = new ArrayList<String>();
        lines.add("1 1:1  3:1  4:1   6:1");
        lines.add("2 ");
        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        try {
            for (String line : lines)
                writer.append(line).append("\n");
        }
        finally {
            writer.close();
        }

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

    @Test(expected = InvalidInputDataException.class)
    public void testReadUnsortedProblem() throws Exception {
        File file = File.createTempFile("svm", "test");
        file.deleteOnExit();

        Collection<String> lines = new ArrayList<String>();
        lines.add("1 1:1  3:1  4:1   6:1");
        lines.add("2 2:1  3:1  5:1   7:1");
        lines.add("1 3:1  5:1  4:1"); // here's the mistake: not correctly sorted

        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        try {
            for (String line : lines)
                writer.append(line).append("\n");
        }
        finally {
            writer.close();
        }

        Train train = new Train();
        train.readProblem(file.getAbsolutePath());
    }


    @Test(expected = InvalidInputDataException.class)
    public void testReadProblemWithInvalidIndex() throws Exception {
        File file = File.createTempFile("svm", "test");
        file.deleteOnExit();

        Collection<String> lines = new ArrayList<String>();
        lines.add("1 1:1  3:1  4:1   6:1");
        lines.add("2 2:1  3:1  5:1  -4:1");

        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        try {
            for (String line : lines)
                writer.append(line).append("\n");
        }
        finally {
            writer.close();
        }

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
        } catch (InvalidInputDataException e) {
            throw e;
        }
    }

    @Test(expected = InvalidInputDataException.class)
    public void testReadWrongProblem() throws Exception {
        File file = File.createTempFile("svm", "test");
        file.deleteOnExit();

        Collection<String> lines = new ArrayList<String>();
        lines.add("1 1:1  3:1  4:1   6:1");
        lines.add("2 2:1  3:1  5:1   7:1");
        lines.add("1 3:1  5:a"); // here's the mistake: incomplete line

        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        try {
            for (String line : lines)
                writer.append(line).append("\n");
        }
        finally {
            writer.close();
        }

        Train train = new Train();
        try {
            train.readProblem(file.getAbsolutePath());
        } catch (InvalidInputDataException e) {
            throw e;
        }
    }
}
