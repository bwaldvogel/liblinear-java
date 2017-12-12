package de.bwaldvogel.liblinear;

class L2R_LrFunction implements Function {

    private final double[] C;
    private final double[] z;
    private final double[] D;
    private final Problem  prob;

    public L2R_LrFunction(Problem prob, double[] C) {
        int l = prob.l;

        this.prob = prob;

        z = new double[l];
        D = new double[l];
        this.C = C;
    }

    private void Xv(double[] v, double[] Xv) {
        int l = prob.l;
        Feature[][] x = prob.x;

        for (int i = 0; i < l; i++)
            Xv[i] = SparseOperator.dot(v, x[i]);
    }

    private void XTv(double[] v, double[] XTv) {
        int l = prob.l;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            XTv[i] = 0;

        for (int i = 0; i < l; i++) {
            SparseOperator.axpy(v[i], x[i], XTv);
        }
    }

    @Override
    public double fun(double[] w) {
        int i;
        double f = 0;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        Xv(w, z);

        for (i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2.0;
        for (i = 0; i < l; i++) {
            double yz = y[i] * z[i];
            if (yz >= 0)
                f += C[i] * Math.log(1 + Math.exp(-yz));
            else
                f += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
        }

        return (f);
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        for (i = 0; i < l; i++) {
            z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
            D[i] = z[i] * (1 - z[i]);
            z[i] = C[i] * (z[i] - 1) * y[i];
        }
        XTv(z, g);

        for (i = 0; i < w_size; i++)
            g[i] = w[i] + g[i];
    }

    @Override
    public void Hv(double[] s, double[] Hs) {
        int i;
        int l = prob.l;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            Hs[i] = 0;
        for (i = 0; i < l; i++) {
            Feature[] xi = x[i];
            double xTs = SparseOperator.dot(s, xi);

            xTs = C[i] * D[i] * xTs;

            SparseOperator.axpy(xTs, xi, Hs);
        }
        for (i = 0; i < w_size; i++)
            Hs[i] = s[i] + Hs[i];
    }

    @Override
    public int get_nr_variable() {
        return prob.n;
    }

    @Override
    public void get_diagH(double[] M) {
        int l = prob.l;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            M[i] = 1;

        for (int i = 0; i < l; i++) {
            for (Feature s : x[i]) {
                M[s.getIndex() - 1] += s.getValue() * s.getValue() * C[i] * D[i];
            }
        }
    }

}
