package liblinear;

import static liblinear.Linear.info;
import static org.netlib.blas.DAXPY.DAXPY;
import static org.netlib.blas.DDOT.DDOT;
import static org.netlib.blas.DNRM2.DNRM2;
import static org.netlib.blas.DSCAL.DSCAL;


class Tron {

    private final Function fun_obj;

    private final double   eps;

    private final int      max_iter;

    public Tron( final Function fun_obj ) {
        this(fun_obj, 0.1);
    }

    public Tron( final Function fun_obj, double eps ) {
        this(fun_obj, eps, 1000);
    }

    public Tron( final Function fun_obj, double eps, int max_iter ) {
        this.fun_obj = fun_obj;
        this.eps = eps;
        this.max_iter = max_iter;
    }

    // void tron(double *w)
    void tron(double[] w) {
        // Parameters for updating the iterates.
        double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

        // Parameters for updating the trust region size delta.
        double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

        int n = fun_obj.get_nr_variable();
        int i, cg_iter;
        double delta, snorm, one = 1.0;
        double alpha, f, fnew, prered, actred, gs;
        int search = 1, iter = 1, inc = 1;
        double[] s = new double[n];
        double[] r = new double[n];
        double[] w_new = new double[n];
        double[] g = new double[n];

        for (i = 0; i < n; i++)
            w[i] = 0;

        f = fun_obj.fun(w);
        fun_obj.grad(w, g);
        delta = DNRM2(n, g, inc);
        // delta = dnrm2_(&n, g, &inc);
        double gnorm1 = delta;
        double gnorm = gnorm1;

        if (gnorm <= eps * gnorm1) search = 0;

        iter = 1;

        while (iter <= max_iter && search != 0) {
            cg_iter = trcg(delta, g, s, r);

            // memcpy(w_new, w, sizeof(double)*n);
            System.arraycopy(w, 0, w_new, 0, n);
            DAXPY(n, one, s, inc, w_new, inc);

            gs = DDOT(n, g, inc, s, inc);
            // gs = ddot_(&n, g, &inc, s, &inc);
            prered = -0.5 * (gs - DDOT(n, s, inc, r, inc));
            fnew = fun_obj.fun(w_new);

            // Compute the actual reduction.
            actred = f - fnew;

            // On the first iteration, adjust the initial step bound.
            snorm = DNRM2(n, s, inc);
            // snorm = dnrm2_(&n, s, &inc);
            if (iter == 1) delta = Math.min(delta, snorm);

            // Compute prediction alpha*snorm of the step.
            if (fnew - f - gs <= 0)
                alpha = sigma3;
            else
                alpha = Math.max(sigma1, -0.5 * (gs / (fnew - f - gs)));

            // Update the trust region bound according to the ratio of actual to
            // predicted reduction.
            if (actred < eta0 * prered)
                delta = Math.min(Math.max(alpha, sigma1) * snorm, sigma2 * delta);
            else if (actred < eta1 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * snorm, sigma2 * delta));
            else if (actred < eta2 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * snorm, sigma3 * delta));
            else
                delta = Math.max(delta, Math.min(alpha * snorm, sigma3 * delta));

            info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d%n", iter, actred, prered, delta, f, gnorm, cg_iter);

            if (actred > eta0 * prered) {
                iter++;
                // memcpy(w, w_new, sizeof(double)*n);
                System.arraycopy(w_new, 0, w, 0, n);
                f = fnew;
                fun_obj.grad(w, g);

                gnorm = DNRM2(n, g, inc);
                // gnorm = dnrm2_(&n, g, &inc);
                if (gnorm <= eps * gnorm1) break;
            }
            if (f < -1.0e+32) {
                info("warning: f < -1.0e+32%n");
                break;
            }
            if (Math.abs(actred) <= 0 && prered <= 0) {
                info("warning: actred and prered <= 0%n");
                break;
            }
            if (Math.abs(actred) <= 1.0e-12 * Math.abs(f) && Math.abs(prered) <= 1.0e-12 * Math.abs(f)) {
                info("warning: actred and prered too small%n");
                break;
            }
        }
    }

    // int TRON::trcg(double delta, double *g, double *s, double *r)
    int trcg(double delta, double[] g, double[] s, double[] r) {
        int i, inc = 1;
        int n = fun_obj.get_nr_variable();
        double one = 1;
        double[] d = new double[n];
        double[] Hd = new double[n];
        double rTr, rnewTrnew, cgtol;

        for (i = 0; i < n; i++) {
            s[i] = 0;
            r[i] = -g[i];
            d[i] = r[i];
        }
        cgtol = 0.1 * DNRM2(n, g, inc);

        int cg_iter = 0;
        // rTr = ddot_(&n, r, &inc, r, &inc);
        rTr = DDOT(n, r, inc, r, inc);

        while (true) {
            if (DNRM2(n, r, inc) <= cgtol) break;
            cg_iter++;
            fun_obj.Hv(d, Hd);

            double alpha = rTr / DDOT(n, d, inc, Hd, inc);
            DAXPY(n, alpha, d, inc, s, inc);
            // daxpy_(&n, &alpha, d, &inc, s, &inc);
            // if (dnrm2_(&n, s, &inc) > delta)
            if (DNRM2(n, s, inc) > delta) {
                info("cg reaches trust region boundary\n");
                alpha = -alpha;
                // daxpy_(&n, &alpha, d, &inc, s, &inc);
                DAXPY(n, alpha, d, inc, s, inc);

                double std = DDOT(n, s, inc, d, inc);
                double sts = DDOT(n, s, inc, s, inc);
                double dtd = DDOT(n, d, inc, d, inc);
                double dsq = delta * delta;
                double rad = Math.sqrt(std * std + dtd * (dsq - sts));
                if (std >= 0)
                    alpha = (dsq - sts) / (std + rad);
                else
                    alpha = (rad - std) / dtd;
                DAXPY(n, alpha, d, inc, s, inc);
                alpha = -alpha;
                DAXPY(n, alpha, Hd, inc, r, inc);
                break;
            }
            alpha = -alpha;
            DAXPY(n, alpha, Hd, inc, r, inc);
            rnewTrnew = DDOT(n, r, inc, r, inc);
            double beta = rnewTrnew / rTr;
            DSCAL(n, beta, d, inc);
            DAXPY(n, one, r, inc, d, inc);
            rTr = rnewTrnew;
        }

        return (cg_iter);
    }

    double norm_inf(int n, double[] x) {
        double dmax = Math.abs(x[0]);
        for (int i = 1; i < n; i++)
            if (Math.abs(x[i]) >= dmax) dmax = Math.abs(x[i]);
        return (dmax);
    }
}
