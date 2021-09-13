package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.*;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;


class PredictTest {

    private final Model         testModel = LinearTest.createRandomModel();
    private final StringBuilder sb        = new StringBuilder();
    private final Writer        writer    = new StringWriter();

    private final ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

    @BeforeEach
    public void setUp() {
        Linear.setDebugOutput(new PrintStream(byteArrayOutputStream));
        assertThat(testModel.getNrClass()).isGreaterThanOrEqualTo(2);
        assertThat(testModel.getNrFeature()).isGreaterThanOrEqualTo(10);
    }

    @AfterEach
    public void tearDown() {
        Linear.enableDebugOutput();
    }

    private void testWithLines(StringBuilder sb) throws Exception {
        try (StringReader stringReader = new StringReader(sb.toString());
             BufferedReader reader = new BufferedReader(stringReader)) {
            Predict.doPredict(reader, writer, testModel, false);
        }
    }

    @Test
    void testDoPredictCorruptLine() throws Exception {
        sb.append(testModel.label[0]).append(" abc").append("\n");

        assertThatExceptionOfType(RuntimeException.class)
            .isThrownBy(() -> testWithLines(sb))
            .withMessage("Wrong input format at line 1");
    }

    @Test
    void testDoPredictCorruptLine2() throws Exception {
        sb.append(testModel.label[0]).append(" 1:").append("\n");

        assertThatExceptionOfType(RuntimeException.class)
            .isThrownBy(() -> testWithLines(sb))
            .withMessage("Can't convert empty string to integer");
    }

    @Test
    void testDoPredict() throws Exception {
        sb.append(testModel.label[0]).append(" 1:0.32393").append("\n");
        sb.append(testModel.label[1]).append(" 2:-71.555   9:88223").append("\n");
        testWithLines(sb);
        assertThat(writer.toString()).isNotEmpty();
    }

    @Test
    void testTrainAndPredict(@TempDir Path tempDir) throws Exception {
        String modelFile = tempDir.resolve("model").toString();
        Train.main(new String[] {"-s", "0", "src/test/datasets/dna.scale/dna.scale", modelFile});

        Path predictionsFile = tempDir.resolve("predictions");
        Predict.main(new String[] {"-b", "1", "src/test/datasets/dna.scale/dna.scale.t", modelFile, predictionsFile.toString()});
        List<String> predictions = Files.readAllLines(predictionsFile, Linear.FILE_CHARSET);
        assertThat(predictions).hasSize(1187);
        assertThat(predictions.get(0)).isEqualTo("labels 3 1 2");

        String loggedString = byteArrayOutputStream.toString();
        assertThat(loggedString).containsPattern("Accuracy = 95[.,]0253% \\(1127/1186\\)");
    }
}
