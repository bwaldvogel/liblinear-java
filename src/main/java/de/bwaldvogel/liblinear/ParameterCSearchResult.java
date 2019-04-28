package de.bwaldvogel.liblinear;

public class ParameterCSearchResult {

    private final double bestC;
    private final double bestScore;

    public ParameterCSearchResult(double bestC, double bestScore) {
        this.bestC = bestC;
        this.bestScore = bestScore;
    }

    public double getBestC() {
        return bestC;
    }

    public double getBestScore() {
        return bestScore;
    }

}
