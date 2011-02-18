package de.bwaldvogel.liblinear;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;

import org.junit.Before;
import org.junit.Test;


public class PredictTest {

    private Model         testModel = LinearTest.createRandomModel();
    private StringBuilder sb        = new StringBuilder();
    private Writer        writer    = new StringWriter();

    @Before
    public void setUp() {
        System.setOut(mock(PrintStream.class)); // dev/null
        assertThat(testModel.getNrClass()).isGreaterThanOrEqualTo(2);
        assertThat(testModel.getNrFeature()).isGreaterThanOrEqualTo(10);
    }

    private void testWithLines(StringBuilder sb) throws Exception {
        BufferedReader reader = new BufferedReader(new StringReader(sb.toString()));

        Predict.doPredict(reader, writer, testModel);
    }

    @Test(expected = RuntimeException.class)
    public void testDoPredictCorruptLine() throws Exception {
        sb.append(testModel.label[0]).append(" abc").append("\n");
        testWithLines(sb);
    }

    @Test(expected = RuntimeException.class)
    public void testDoPredictCorruptLine2() throws Exception {
        sb.append(testModel.label[0]).append(" 1:").append("\n");
        testWithLines(sb);
    }

    @Test
    public void testDoPredict() throws Exception {
        sb.append(testModel.label[0]).append(" 1:0.32393").append("\n");
        sb.append(testModel.label[1]).append(" 2:-71.555   9:88223").append("\n");
        testWithLines(sb);
        assertThat(writer.toString()).isNotEmpty();
    }
}
