package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.Linear.NL;
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
            train.parse_command_line(new String[] {"-B", "5.3", "-s", "" + solver.ordinal(), "model-filename"});
            Parameter param = train.getParameter();
            assertThat(param.solverType).isEqualTo(solver);
            // check default eps
            if (solver.ordinal() == 0 || solver.ordinal() == 2 //
                || solver.ordinal() == 5 || solver.ordinal() == 6) {
                assertThat(param.eps).isEqualTo(0.01);
            } else {
                assertThat(param.eps).isEqualTo(0.1);
            }
            // check if bias is set
            assertThat(train.getBias()).isEqualTo(5.3);
        }
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
                writer.append(line).append(NL);
        }
        finally {
            writer.close();
        }

        Train train = new Train();
        train.readProblem(file.getAbsolutePath());

        Problem prob = train.getProblem();
        assertThat(prob.bias).isEqualTo(1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new int[] {1, 2, 1, 1, 2});
        assertThat(prob.n).isEqualTo(8);
        assertThat(prob.l).isEqualTo(prob.y.length);
        assertThat(prob.x).hasSize(prob.y.length);

        for (FeatureNode[] nodes : prob.x) {

            assertThat(nodes.length).isLessThanOrEqualTo(prob.n);
            for (FeatureNode node : nodes) {
                // bias term
                if (prob.bias >= 0 && nodes[nodes.length - 1] == node) {
                    assertThat(node.index).isEqualTo(prob.n);
                    assertThat(node.value).isEqualTo(prob.bias);
                } else {
                    assertThat(node.index).isLessThan(prob.n);
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
                writer.append(line).append(NL);
        }
        finally {
            writer.close();
        }

        Problem prob = Train.readProblem(file, -1.0);
        assertThat(prob.bias).isEqualTo(-1);
        assertThat(prob.y).hasSize(lines.size());
        assertThat(prob.y).isEqualTo(new int[] {1, 2});
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
                writer.append(line).append(NL);
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
                writer.append(line).append(NL);
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
                writer.append(line).append(NL);
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
