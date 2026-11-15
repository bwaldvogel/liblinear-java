package de.bwaldvogel.liblinear;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;


final class TestUtils {

    private TestUtils() {
    }

    static void writeToFile(Path file, List<String> lines) throws IOException {
        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            for (String line : lines) {
                bufferedWriter.append(line).append("\n");
            }
            bufferedWriter.flush();
        }
    }
}
