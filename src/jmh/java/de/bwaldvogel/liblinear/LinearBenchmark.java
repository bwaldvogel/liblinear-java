package de.bwaldvogel.liblinear;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

@Fork(1)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2, time = 5, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
public class LinearBenchmark {

    @Benchmark
    public void readProblem(DatasetParameters datasetParameters) throws Exception {
        Path trainingFile = getTrainingFile(datasetParameters.dataset);
        try (InputStream inputStream = getInputStream(trainingFile)) {
            Train.readProblem(inputStream, -1);
        }
    }

    @Benchmark
    public void train(BenchmarkParameters benchmarkParameters) {
        Linear.disableDebugOutput();
        Linear.train(benchmarkParameters.problem, new Parameter(benchmarkParameters.solverType, 1, 1e-3));
    }

    public enum Dataset {
        RCV1, Splice, DnaScale
    }


    @State(Scope.Benchmark)
    public static class BenchmarkParameters {

        @Param
        private Dataset dataset;

        @Param({"L2R_LR", "L2R_L2LOSS_SVC_DUAL"})
        private SolverType solverType;

        private Problem problem;

        @Setup
        public void loadDataset() throws Exception {
            Path trainingFile = getTrainingFile(dataset);
            try (InputStream inputStream = getInputStream(trainingFile)) {
                problem = Train.readProblem(inputStream, -1);
            }
        }

    }

    @State(Scope.Benchmark)
    public static class DatasetParameters {

        @Param
        private Dataset dataset;

    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(".*" + LinearBenchmark.class.getSimpleName() + ".*")
                .build();

        new Runner(opt).run();
    }

    private static InputStream getInputStream(Path path) throws IOException {
        InputStream inputStream = Files.newInputStream(path);
        if (path.toString().endsWith(".bz2")) {
            return new BZip2CompressorInputStream(inputStream);
        }
        return inputStream;
    }

    private static Path getTrainingFile(Dataset dataset) {
        Path datasetDirectory = Paths.get(System.getProperty("dataset.directory", "src/test/datasets"));
        switch (dataset) {
            case RCV1:
                return datasetDirectory.resolve("rcv1").resolve("rcv1_train.binary.bz2");
            case Splice:
                return datasetDirectory.resolve("splice").resolve("splice");
            case DnaScale:
                return datasetDirectory.resolve("dna.scale").resolve("dna.scale");
            default:
                throw new IllegalArgumentException("Unknown dataset: " + dataset);
        }
    }

}
