package de.bwaldvogel.liblinear;

class L2R_L2_SvcFunction implements Function {

    protected final Problem  prob;
    protected final double[] C;
    protected final int[]    I;
    protected final double[] z;

    protected int            sizeI;

    public L2R_L2_SvcFunction(Problem prob, double[] C) {
        int l = prob.l;

        this.prob = prob;

        z = new double[l];
        I = new int[l];
        this.C = C;
    }

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
            z[i] = y[i] * z[i];
            double d = 1 - z[i];
            if (d > 0) f += C[i] * d * d;
        }

        return (f);
    }

    public int get_nr_variable() {
        return prob.n;
    }

    public void grad(double[] w, double[] g) {
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        sizeI = 0;
        for (int i = 0; i < l; i++) {
            if (z[i] < 1) {
                z[sizeI] = C[i] * y[i] * (z[i] - 1);
                I[sizeI] = i;
                sizeI++;
            }
        }
        subXTv(z, g);

        for (int i = 0; i < w_size; i++)
            g[i] = w[i] + 2 * g[i];
    }

    public void Hv(double[] s, double[] Hs) {
        int i;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            Hs[i] = 0;
        for (i = 0; i < sizeI; i++) {
            Feature[] xi = x[I[i]];
            double xTs = SparseOperator.dot(s, xi);
            xTs = C[I[i]] * xTs;

            SparseOperator.axpy(xTs, xi, Hs);
        }
        for (i = 0; i < w_size; i++)
            Hs[i] = s[i] + 2 * Hs[i];
    }

    protected void subXTv(double[] v, double[] XTv) {
        int i;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            XTv[i] = 0;
        for (i = 0; i < sizeI; i++)
            SparseOperator.axpy(v[i], x[I[i]], XTv);
    }

    protected void Xv(double[] v, double[] Xv) {
        int l = prob.l;
        Feature[][] x = prob.x;

        for (int i = 0; i < l; i++)
            Xv[i] = SparseOperator.dot(v, x[i]);
    }

}
