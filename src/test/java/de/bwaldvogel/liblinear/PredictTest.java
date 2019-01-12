package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;


public class PredictTest {

    private Model         testModel = LinearTest.createRandomModel();
    private StringBuilder sb        = new StringBuilder();
    private Writer        writer    = new StringWriter();

    private ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Before
    public void setUp() {
        Linear.resetRandom();
        Linear.setDebugOutput(new PrintStream(byteArrayOutputStream));
        assertThat(testModel.getNrClass()).isGreaterThanOrEqualTo(2);
        assertThat(testModel.getNrFeature()).isGreaterThanOrEqualTo(10);
    }

    @After
    public void tearDown() {
        Linear.enableDebugOutput();
    }

    private void testWithLines(StringBuilder sb) throws Exception {
        try (StringReader stringReader = new StringReader(sb.toString());
             BufferedReader reader = new BufferedReader(stringReader)) {
            Predict.doPredict(reader, writer, testModel);
        }
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

    @Test
    public void testTrainAndPredict() throws Exception {
        String modelFile = temporaryFolder.newFile("model").toString();
        Train.main(new String[] {"-s", "0", "src/test/datasets/dna.scale/dna.scale", modelFile});

        File predictionsFile = temporaryFolder.newFile("predictions");
        Predict.main(new String[] {"-b", "1", "src/test/datasets/dna.scale/dna.scale.t", modelFile, predictionsFile.toString()});
        List<String> predictions = Files.readAllLines(predictionsFile.toPath(), Linear.FILE_CHARSET);
        assertThat(predictions).hasSize(1187);
        assertThat(predictions.get(0)).isEqualTo("labels 3 1 2");

        String loggedString = byteArrayOutputStream.toString();
        assertThat(loggedString).contains("Accuracy = 94.9410% (1126/1186)");
    }
}
