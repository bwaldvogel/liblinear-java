package de.bwaldvogel.liblinear;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.List;


final class TestUtils {

    private TestUtils() {
    }

    static void writeToFile(File file, List<String> lines) throws IOException {
        try (Writer writer = new FileWriter(file);
             BufferedWriter bufferedWriter = new BufferedWriter(writer)) {
            for (String line : lines) {
                bufferedWriter.append(line).append("\n");
            }
            bufferedWriter.flush();
        }
    }

    static String repeat(String stringToRepeat, int numTimes) {
        StringBuilder longString = new StringBuilder();
        for (int i = 0; i < numTimes; i++) {
            longString.append(stringToRepeat);
        }
        return longString.toString();
    }
}
