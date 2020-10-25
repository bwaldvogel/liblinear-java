package de.bwaldvogel.liblinear;

/**
 * @since 1.91
 */
public class L2R_L2_SvrFunction extends L2R_L2_SvcFunction {

    private final double p;

    public L2R_L2_SvrFunction(Problem prob, Parameter param, double[] C) {
        super(prob, param, C);
        this.p = param.p;
    }

    @Override
    protected double C_times_loss(int i, double wx_i) {
        double d = wx_i - prob.y[i];
        if (d < -p)
            return C[i] * (d + p) * (d + p);
        else if (d > p)
            return C[i] * (d - p) * (d - p);
        return 0;
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();
        double d;

        sizeI = 0;
        for (i = 0; i < l; i++) {
            d = wx[i] - y[i];

            // generate index set I
            if (d < -p) {
                tmp[sizeI] = C[i] * (d + p);
                I[sizeI] = i;
                sizeI++;
            } else if (d > p) {
                tmp[sizeI] = C[i] * (d - p);
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

}
