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

    static double sparse_dot(Feature[] x1, Feature[] x2) {
        double ret = 0;

        int pos1 = 0;
        int pos2 = 0;

        while (pos1 < x1.length && pos2 < x2.length) {
            int index1 = x1[pos1].getIndex();
            int index2 = x2[pos2].getIndex();
            if (index1 == index2) {
                ret += x1[pos1].getValue() * x2[pos2].getValue();
                pos1++;
                pos2++;
            } else {
                if (index1 > index2) {
                    pos2++;
                } else {
                    pos1++;
                }
            }
        }

        return (ret);
    }

    static void axpy(double a, Feature[] x, double[] y) {
        for (Feature feature : x) {
            y[feature.getIndex() - 1] += a * feature.getValue();
        }
    }

}
