package de.bwaldvogel.liblinear;

class L2R_L2_SvcFunction extends L2R_ErmFunction {

    protected final int[] I;
    protected       int   sizeI;

    public L2R_L2_SvcFunction(Problem prob, Parameter param, double[] C) {
        super(prob, param, C);
        I = new int[prob.l];
    }

    @Override
    protected double C_times_loss(int i, double wx_i) {
        double d = 1 - prob.y[i] * wx_i;
        if (d > 0)
            return C[i] * d * d;
        else
            return 0;
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        sizeI = 0;
        for (i = 0; i < l; i++) {
            tmp[i] = wx[i] * y[i];
            if (tmp[i] < 1) {
                tmp[sizeI] = C[i] * y[i] * (tmp[i] - 1);
                I[sizeI] = i;
                sizeI++;
            }
        }
        subXTv(tmp, g);

        for (i = 0; i < w_size; i++)
            g[i] = w[i] + 2 * g[i];
        if (!regularize_bias)
            g[w_size - 1] -= w[w_size - 1];
    }

    @Override
    public void get_diag_preconditioner(double[] M) {
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            M[i] = 1;
        if (!regularize_bias)
            M[w_size - 1] = 0;

        for (int i = 0; i < sizeI; i++) {
            int idx = I[i];
            for (Feature s : x[idx]) {
                M[s.getIndex() - 1] += s.getValue() * s.getValue() * C[idx] * 2;
            }
        }
    }

    @Override
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
        if (!regularize_bias)
            Hs[w_size - 1] -= s[w_size - 1];
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

}
