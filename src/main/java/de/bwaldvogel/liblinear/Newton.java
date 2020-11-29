package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.Linear.*;


class Newton {

    private final Function fun_obj;
    private final double   eps;
    private final int      max_iter;
    private final double   eps_cg;

    Newton(Function fun_obj, double eps, int max_iter) {
        this(fun_obj, eps, max_iter, 0.5);
    }

    Newton(Function fun_obj, double eps, int max_iter, double eps_cg) {
        this.fun_obj = fun_obj;
        this.eps = eps;
        this.max_iter = max_iter;
        this.eps_cg = eps_cg;
    }

    void newton(double[] w) {
        int n = fun_obj.get_nr_variable();
        int i, cg_iter;
        double step_size;
        double f, fold, actred;
        double init_step_size = 1;
        boolean search = true;
        int iter = 1;
        MutableInt inc = new MutableInt(1);
        double[] s = new double[n];
        double[] r = new double[n];
        double[] g = new double[n];

        final double alpha_pcg = 0.01;
        double[] M = new double[n];

        // calculate gradient norm at w=0 for stopping condition.
        double[] w0 = new double[n];
        for (i = 0; i < n; i++)
            w0[i] = 0;
        fun_obj.fun(w0);
        fun_obj.grad(w0, g);

        double gnorm0 = Blas.dnrm2_(n, g, inc);

        f = fun_obj.fun(w);
        fun_obj.grad(w, g);
        double gnorm = Blas.dnrm2_(n, g, inc);
        info("init f %5.3e |g| %5.3e%n", f, gnorm);

        if (gnorm <= eps * gnorm0)
            search = false;

        while (iter <= max_iter && search) {
            fun_obj.get_diag_preconditioner(M);
            for (i = 0; i < n; i++)
                M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
            cg_iter = pcg(g, M, s, r);

            fold = f;
            MutableDouble fReference = new MutableDouble(f);
            step_size = fun_obj.linesearch_and_update(w, s, fReference, g, init_step_size);
            f = fReference.get();

            if (step_size == 0) {
                info("WARNING: line search fails%n");
                break;
            }

            fun_obj.grad(w, g);
            gnorm = Blas.dnrm2_(n, g, inc);

            info("iter %2d f %5.3e |g| %5.3e CG %3d step_size %4.2e%n", iter, f, gnorm, cg_iter, step_size);

            if (gnorm <= eps * gnorm0)
                break;
            if (f < -1.0e+32) {
                info("WARNING: f < -1.0e+32%n");
                break;
            }
            actred = fold - f;
            if (Math.abs(actred) <= 1.0e-12 * Math.abs(f)) {
                info("WARNING: actred too small%n");
                break;
            }

            iter++;
        }

        if (iter >= max_iter)
            info("%nWARNING: reaching max number of Newton iterations%n");
    }

    private int pcg(double[] g, double[] M, double[] s, double[] r) {
        int i, inc = 1;
        int n = fun_obj.get_nr_variable();
        double one = 1;
        double[] d = new double[n];
        double[] Hd = new double[n];
        double zTr, znewTrnew, alpha, beta, cgtol, dHd;
        double[] z = new double[n];
        double Q = 0, newQ, Qdiff;

        for (i = 0; i < n; i++) {
            s[i] = 0;
            r[i] = -g[i];
            z[i] = r[i] / M[i];
            d[i] = z[i];
        }

        zTr = Blas.ddot_(n, z, inc, r, inc);
        double gMinv_norm = Math.sqrt(zTr);
        cgtol = Math.min(eps_cg, Math.sqrt(gMinv_norm));
        int cg_iter = 0;
        int max_cg_iter = Math.max(n, 5);

        while (cg_iter < max_cg_iter) {
            cg_iter++;

            fun_obj.Hv(d, Hd);
            dHd = Blas.ddot_(n, d, inc, Hd, inc);
            // avoid 0/0 in getting alpha
            if (dHd <= 1.0e-16)
                break;

            alpha = zTr / dHd;
            Blas.daxpy_(n, alpha, d, inc, s, inc);
            alpha = -alpha;
            Blas.daxpy_(n, alpha, Hd, inc, r, inc);

            // Using quadratic approximation as CG stopping criterion
            newQ = -0.5 * (Blas.ddot_(n, s, inc, r, inc) - Blas.ddot_(n, s, inc, g, inc));
            Qdiff = newQ - Q;
            if (newQ <= 0 && Qdiff <= 0) {
                if (cg_iter * Qdiff >= cgtol * newQ)
                    break;
            } else {
                info("WARNING: quadratic approximation > 0 or increasing in CG%n");
                break;
            }
            Q = newQ;

            for (i = 0; i < n; i++)
                z[i] = r[i] / M[i];
            znewTrnew = Blas.ddot_(n, z, inc, r, inc);
            beta = znewTrnew / zTr;
            Blas.dscal_(n, beta, d, inc);
            Blas.daxpy_(n, one, z, inc, d, inc);
            zTr = znewTrnew;
        }

        if (cg_iter == max_cg_iter)
            info("WARNING: reaching maximal number of CG steps%n");

        return (cg_iter);
    }

}
