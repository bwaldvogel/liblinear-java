package de.bwaldvogel.liblinear;

public class ParameterSearchResult {

    private final double bestC;
    private final double bestRate;

    public ParameterSearchResult(double bestC, double bestRate) {
        this.bestC = bestC;
        this.bestRate = bestRate;
    }

    public double getBestC() {
        return bestC;
    }

    public double getBestRate() {
        return bestRate;
    }

}
