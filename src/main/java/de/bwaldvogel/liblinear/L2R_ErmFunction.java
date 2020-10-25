package de.bwaldvogel.liblinear;

// L2-regularized empirical risk minimization
// min_w w^Tw/2 + \sum C_i \xi(w^Tx_i), where \xi() is the loss
abstract class L2R_ErmFunction implements Function {

    final double[] C;
    final Problem  prob;
    double[] wx;
    double[] tmp; // a working array
    private double  wTw;
    final   boolean regularize_bias;

    L2R_ErmFunction(Problem prob, Parameter parameter, double[] C) {
        int l = prob.l;

        this.prob = prob;

        wx = new double[l];
        tmp = new double[l];
        this.C = C;
        this.regularize_bias = parameter.regularize_bias;
    }

    void Xv(double[] v, double[] Xv) {
        int i;
        int l = prob.l;
        Feature[][] x = prob.x;

        for (i = 0; i < l; i++)
            Xv[i] = SparseOperator.dot(v, x[i]);
    }

    void XTv(double[] v, double[] XTv) {
        int l = prob.l;
        int w_size = get_nr_variable();
        Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            XTv[i] = 0;

        for (int i = 0; i < l; i++) {
            SparseOperator.axpy(v[i], x[i], XTv);
        }
    }

    protected abstract double C_times_loss(int i, double wx_i);

    @Override
    public double fun(double[] w) {
        int i;
        double f = 0;
        int l = prob.l;
        int w_size = get_nr_variable();

        wTw = 0;
        Xv(w, wx);

        for (i = 0; i < w_size; i++)
            wTw += w[i] * w[i];
        if (!regularize_bias)
            wTw -= w[w_size - 1] * w[w_size - 1];
        for (i = 0; i < l; i++)
            f += C_times_loss(i, wx[i]);
        f = f + 0.5 * wTw;

        return (f);
    }

    @Override
    public int get_nr_variable() {
        return prob.n;
    }

    // On entry *f must be the function value of w
    // On exit w is updated and *f is the new function value
    @Override
    public double linesearch_and_update(double[] w, double[] s, MutableDouble f, double[] g, double alpha) {
        int i;
        int l = prob.l;
        double sTs = 0;
        double wTs = 0;
        double gTs = 0;
        double eta = 0.01;
        int w_size = get_nr_variable();
        int max_num_linesearch = 20;
        double fold = f.get();
        Xv(s, tmp);

        for (i = 0; i < w_size; i++) {
            sTs += s[i] * s[i];
            wTs += s[i] * w[i];
            gTs += s[i] * g[i];
        }
        if (!regularize_bias) {
            // bias not used in calculating (w + \alpha s)^T (w + \alpha s)
            sTs -= s[w_size - 1] * s[w_size - 1];
            wTs -= s[w_size - 1] * w[w_size - 1];
        }

        int num_linesearch = 0;
        for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
            double loss = 0;
            for (i = 0; i < l; i++) {
                double inner_product = tmp[i] * alpha + wx[i];
                loss += C_times_loss(i, inner_product);
            }
            f.set(loss + (alpha * alpha * sTs + wTw) / 2.0 + alpha * wTs);
            if (f.get() - fold <= eta * alpha * gTs) {
                for (i = 0; i < l; i++)
                    wx[i] += alpha * tmp[i];
                break;
            } else
                alpha *= 0.5;
        }

        if (num_linesearch >= max_num_linesearch) {
            f.set(fold);
            return 0;
        } else
            for (i = 0; i < w_size; i++)
                w[i] += alpha * s[i];

        wTw += alpha * alpha * sTs + 2 * alpha * wTs;
        return alpha;
    }

}
