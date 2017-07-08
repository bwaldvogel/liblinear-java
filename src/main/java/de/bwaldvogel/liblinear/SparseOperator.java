package de.bwaldvogel.liblinear;

class SparseOperator {

    static double nrm2_sq(Feature[] x) {
        double ret = 0;
        for (Feature feature : x) {
            ret += feature.getValue() * feature.getValue();
        }
        return (ret);
    }

    static double dot(double[] s, Feature[] x) {
        double ret = 0;
        for (Feature feature : x) {
            ret += s[feature.getIndex() - 1] * feature.getValue();
        }
        return (ret);
    }

    static void axpy(double a, Feature[] x, double[] y) {
        for (Feature feature : x) {
            y[feature.getIndex() - 1] += a * feature.getValue();
        }
    }

}
