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

    private void Xv(final double[] v, final double[] Xv) {
        int l = prob.l;
        final Feature[][] x = prob.x;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                @Override
                public void run(int start, int endExclusive) {
                    for (int i = start; i < endExclusive; i++) {
                        Xv[i] = SparseOperator.dot(v, x[i]);
                    }
                }
            }, l);
        } else {
            for (int i = 0; i < l; i++) {
                Xv[i] = SparseOperator.dot(v, x[i]);
            }
        }
    }

    private void XTv(final double[] v, final double[] XTv) {
        int l = prob.l;
        int w_size = get_nr_variable();
        final Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            XTv[i] = 0;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                private final Object xtvSynchronizer = new Object();
                private ThreadLocalDoubleArray threadLocalResult = new ThreadLocalDoubleArray(XTv.length);

                @Override
                public void run(int start, int endExclusive) {
                    double[] res = threadLocalResult.get();
                    for (int i = start; i < endExclusive; i++) {
                        SparseOperator.axpy(v[i], x[i], res);
                    }

                    synchronized (xtvSynchronizer) {
                        for (int i = 0; i < XTv.length; i++) {
                            XTv[i] += res[i];
                        }
                    }
                }
            }, l);
        } else {
            for (int i = 0; i < l; i++) {
                SparseOperator.axpy(v[i], x[i], XTv);
            }
        }
    }

    @Override
    public double fun(final double[] w) {
        int i;
        double f = 0;
        final double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        Xv(w, z);

        for (i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2.0;

        if (this.threadPool != null) {
            final double[] accumulator = new double[] { f };
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                @Override
                public void run(int start, int endExclusive) {
                    double localF = 0;
                    for (int i = start; i < endExclusive; i++) {
                        double yz = y[i] * z[i];
                        if (yz >= 0) localF += C[i] * Math.log(1 + Math.exp(-yz));
                        else localF += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
                    }
                    synchronized (accumulator) {
                        accumulator[0] += localF;
                    }
                }
            }, l);
            return accumulator[0];
        } else {
            for (i = 0; i < l; i++) {
                double yz = y[i] * z[i];
                if (yz >= 0) f += C[i] * Math.log(1 + Math.exp(-yz));
                else f += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
            }
            return (f);
        }
    }

    @Override
    public void grad(double[] w, double[] g) {
        int i;
        final double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                @Override
                public void run(int start, int endExclusive) {
                    for (int i = start; i < endExclusive; i++) {
                        z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
                        D[i] = z[i] * (1 - z[i]);
                        z[i] = C[i] * (z[i] - 1) * y[i];
                    }
                }
            }, l);
        } else {
            for (i = 0; i < l; i++) {
                z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
                D[i] = z[i] * (1 - z[i]);
                z[i] = C[i] * (z[i] - 1) * y[i];
            }
        }

        XTv(z, g);

        for (i = 0; i < w_size; i++)
            g[i] = w[i] + g[i];
    }

    @Override
    public void Hv(final double[] s, final double[] Hs) {
        int i;
        int l = prob.l;
        int w_size = get_nr_variable();
        final Feature[][] x = prob.x;

        for (i = 0; i < w_size; i++)
            Hs[i] = 0;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                private final Object hsSynchronizer = new Object();
                private ThreadLocalDoubleArray threadLocalResult = new ThreadLocalDoubleArray(Hs.length);

                @Override
                public void run(int start, int endExclusive) {
                    double[] res = threadLocalResult.get();

                    for (int i = start; i < endExclusive; i++) {
                        Feature[] xi = x[i];
                        double xTs = SparseOperator.dot(s, xi);

                        xTs = C[i] * D[i] * xTs;

                        SparseOperator.axpy(xTs, xi, res);
                    }

                    synchronized (hsSynchronizer) {
                        for (int i = 0; i < Hs.length; i++) {
                            Hs[i] += res[i];
                        }
                    }
                }
            }, l);
        } else {
            for (i = 0; i < l; i++) {
                Feature[] xi = x[i];
                double xTs = SparseOperator.dot(s, xi);

                xTs = C[i] * D[i] * xTs;

                SparseOperator.axpy(xTs, xi, Hs);
            }
        }
        for (i = 0; i < w_size; i++)
            Hs[i] = s[i] + Hs[i];
    }

    @Override
    public int get_nr_variable() {
        return prob.n;
    }

    @Override
    public void get_diagH(final double[] M) {
        int l = prob.l;
        int w_size = get_nr_variable();
        final Feature[][] x = prob.x;

        for (int i = 0; i < w_size; i++)
            M[i] = 1;

        if (this.threadPool != null) {
            this.threadPool.execute(new LLThreadPool.RangeConsumer() {
                private final Object mSynchronizer = new Object();
                private ThreadLocalDoubleArray threadLocalResult = new ThreadLocalDoubleArray(M.length);

                @Override
                public void run(int start, int endExclusive) {
                    double[] res = threadLocalResult.get();

                    for (int i = start; i < endExclusive; i++) {
                        for (Feature s : x[i]) {
                            res[s.getIndex() - 1] += s.getValue() * s.getValue() * C[i] * D[i];
                        }
                    }

                    synchronized (mSynchronizer) {
                        for (int i = 0; i < M.length; i++) {
                            M[i] += res[i];
                        }
                    }
                }
            }, l);
        } else {
            for (int i = 0; i < l; i++) {
                for (Feature s : x[i]) {
                    M[s.getIndex() - 1] += s.getValue() * s.getValue() * C[i] * D[i];
                }
            }
        }
    }

}
