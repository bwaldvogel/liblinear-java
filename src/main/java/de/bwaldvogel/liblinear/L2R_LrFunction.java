package de.bwaldvogel.liblinear;

class L2R_LrFunction extends L2R_ErmFunction {

    private final double[] D;

    L2R_LrFunction(Problem prob, Parameter param, double[] C) {
        super(prob, param, C);
        int l = prob.l;
        D = new double[l];
    }

    @Override
    protected double C_times_loss(int i, double wx_i) {
        double ywx_i = wx_i * prob.y[i];
        if (ywx_i >= 0)
            return C[i] * Math.log(1 + Math.exp(-ywx_i));
        else
            return C[i] * (-ywx_i + Math.log(1 + Math.exp(ywx_i)));
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        for (i = 0; i < l; i++) {
            tmp[i] = 1 / (1 + Math.exp(-y[i] * wx[i]));
            D[i] = tmp[i] * (1 - tmp[i]);
            tmp[i] = C[i] * (tmp[i] - 1) * y[i];
        }
        XTv(tmp, g);

        for (i = 0; i < w_size; i++)
            g[i] = w[i] + g[i];
        if (!regularize_bias)
            g[w_size - 1] -= w[w_size - 1];
    }

    @Override
    public void get_diag_preconditioner(double[] M) {
        int l = prob.l;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            M[i] = 1;
        if (!regularize_bias)
            M[w_size - 1] = 0;

        for (int i = 0; i < l; i++) {
            for (Feature xi : x[i]) {
                M[xi.getIndex() - 1] += xi.getValue() * xi.getValue() * C[i] * D[i];
            }
        }
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
        if (!regularize_bias)
            Hs[w_size - 1] -= s[w_size - 1];
    }

}
