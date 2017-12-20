package de.bwaldvogel.liblinear;

class L2R_LrFunction implements Function {

    private final double[] C;
    private final double[] z;
    private final double[] D;
    private final Problem  prob;
    private final LLThreadPool threadPool;

    public L2R_LrFunction(Problem prob, double[] C, LLThreadPool threadPool) {
        int l = prob.l;

        this.prob = prob;

        z = new double[l];
        D = new double[l];
        this.C = C;

        this.threadPool = threadPool; // may be null
    }

    private void Xv_loop(int start, int endExclusive, double[] v, double[] xv) {
        final Feature[][] x = prob.x;

        for (int i = start; i < endExclusive; i++) {
            xv[i] = SparseOperator.dot(v, x[i]);
        }
    }

    private void Xv(final double[] v, final double[] Xv) {
        int l = prob.l;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                @Override
                public void run(int start, int endExclusive) {
                    Xv_loop(start, endExclusive, v, Xv);
                }
            }, l);
        } else {
            Xv_loop(0, l, v, Xv);
        }
    }

    private void XTv_loop(int start, int endExclusive, double[] v, double[] xtv) {
        final Feature[][] x = prob.x;

        for (int i = start; i < endExclusive; i++) {
            SparseOperator.axpy(v[i], x[i], xtv);
        }
    }

    private void XTv(final double[] v, final double[] XTv) {
        int l = prob.l;
        int w_size = get_nr_variable();

        for (int i = 0; i < w_size; i++)
            XTv[i] = 0;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumerWithAccumulatorArray(XTv) {
                @Override
                public void run(int start, int endExclusive, double[] accumulator) {
                    XTv_loop(start, endExclusive, v, accumulator);
                }
            }, l);
        } else {
            XTv_loop(0, l, v, XTv);
        }
    }

    private double fun_loop(int start, int endExclusive) {
        double f = 0;
        double[] y = prob.y;

        for (int i = start; i < endExclusive; i++) {
            double yz = y[i] * z[i];
            if (yz >= 0) f += C[i] * Math.log(1 + Math.exp(-yz));
            else f += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
        }

        return f;
    }

    @Override
    public double fun(final double[] w) {
        int i;
        double f = 0;
        int l = prob.l;
        int w_size = get_nr_variable();

        Xv(w, z);

        for (i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2.0;

        if (this.threadPool != null) {
            final double[] fAcc = new double[] { f };
            this.threadPool.execute(new LLThreadPool.RangeConsumerWithAccumulatorArray(fAcc) {
                @Override
                public void run(int start, int endExclusive, double[] accumulator) {
                    accumulator[0] = fun_loop(start, endExclusive);
                }
            }, l);
            return fAcc[0];
        } else {
            return f + fun_loop(0, l);
        }
    }

    private void grad_loop(int start, int endExclusive) {
        double[] y = prob.y;

        for (int i = start; i < endExclusive; i++) {
            z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
            D[i] = z[i] * (1 - z[i]);
            z[i] = C[i] * (z[i] - 1) * y[i];
        }
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        int l = prob.l;
        int w_size = get_nr_variable();

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                @Override
                public void run(int start, int endExclusive) {
                    grad_loop(start, endExclusive);
                }
            }, l);
        } else {
            grad_loop(0, l);
        }

        XTv(z, g);

        for (i = 0; i < w_size; i++)
            g[i] = w[i] + g[i];
    }

    private void Hv_loop(int start, int endExclusive, double[] s, double[] hs) {
        Feature[][] x = prob.x;

        for (int i = start; i < endExclusive; i++) {
            Feature[] xi = x[i];
            double xTs = SparseOperator.dot(s, xi);

            xTs = C[i] * D[i] * xTs;

            SparseOperator.axpy(xTs, xi, hs);
        }
    }

    @Override
    public void Hv(final double[] s, final double[] Hs) {
        int i;
        int l = prob.l;
        int w_size = get_nr_variable();

        for (i = 0; i < w_size; i++)
            Hs[i] = 0;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumerWithAccumulatorArray(Hs) {
                @Override
                public void run(int start, int endExclusive, double[] hs) {
                    Hv_loop(start, endExclusive, s, hs);
                }
            }, l);
        } else {
            Hv_loop(0, l, s, Hs);
        }
        for (i = 0; i < w_size; i++)
            Hs[i] = s[i] + Hs[i];
    }

    @Override
    public int get_nr_variable() {
        return prob.n;
    }

    private void get_diagH_loop(int start, int endExclusive, double[] m) {
        Feature[][] x = prob.x;

        for (int i = start; i < endExclusive; i++) {
            for (Feature s : x[i]) {
                m[s.getIndex() - 1] += s.getValue() * s.getValue() * C[i] * D[i];
            }
        }
    }

    @Override
    public void get_diagH(final double[] M) {
        int l = prob.l;
        int w_size = get_nr_variable();

        for (int i = 0; i < w_size; i++)
            M[i] = 1;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumerWithAccumulatorArray(M) {
                @Override
                public void run(int start, int endExclusive, double[] m) {
                    get_diagH_loop(start, endExclusive, M);
                }
            }, l);
        } else {
            get_diagH_loop(0, l, M);
        }
    }

}
