package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.Linear.info;

import java.util.concurrent.atomic.AtomicBoolean;


/**
 * Trust Region Newton Method optimization
 */
class Tron {

    private final Function fun_obj;
    private final double   eps;
    private final int      max_iter;
    private final double   eps_cg;

    public Tron(Function fun_obj, double eps, int max_iter, double eps_cg) {
        this.fun_obj = fun_obj;
        this.eps = eps;
        this.max_iter = max_iter;
        this.eps_cg = eps_cg;
    }

    void tron(double[] w) {
        // Parameters for updating the iterates.
        double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

        // Parameters for updating the trust region size delta.
        double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

        int n = fun_obj.get_nr_variable();
        int i, cg_iter;
        double delta = 0, sMnorm, one = 1.0;
        double alpha, f, fnew, prered, actred, gs;
        int search = 1, iter = 1;
        double[] s = new double[n];
        double[] r = new double[n];
        double[] g = new double[n];

        double alpha_pcg = 0.01;
        double[] M = new double[n];

        // calculate gradient norm at w=0 for stopping condition.
        double[] w0 = new double[n];
        for (i = 0; i < n; i++)
            w0[i] = 0;
        fun_obj.fun(w0);
        fun_obj.grad(w0, g);
        double gnorm0 = euclideanNorm(g);

        f = fun_obj.fun(w);
        fun_obj.grad(w, g);
        double gnorm = euclideanNorm(g);

        if (gnorm <= eps * gnorm0)
            search = 0;

        iter = 1;

        double[] w_new = new double[n];
        AtomicBoolean reach_boundary = new AtomicBoolean();
        while (iter <= max_iter && search != 0) {
            fun_obj.get_diagH(M);
            for (i = 0; i < n; i++)
                M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
            if (iter == 1)
                delta = Math.sqrt(uTMv(n, g, M, g));
            cg_iter = trpcg(delta, g, M, s, r, reach_boundary);

            System.arraycopy(w, 0, w_new, 0, n);
            daxpy(one, s, w_new);

            gs = dot(g, s);
            prered = -0.5 * (gs - dot(s, r));
            fnew = fun_obj.fun(w_new);

            // Compute the actual reduction.
            actred = f - fnew;

            // On the first iteration, adjust the initial step bound.
            sMnorm = Math.sqrt(uTMv(n, s, M, s));
            if (iter == 1)
                delta = Math.min(delta, sMnorm);

            // Compute prediction alpha*sMnorm of the step.
            if (fnew - f - gs <= 0)
                alpha = sigma3;
            else
                alpha = Math.max(sigma1, -0.5 * (gs / (fnew - f - gs)));

            // Update the trust region bound according to the ratio of actual to
            // predicted reduction.
            if (actred < eta0 * prered)
                delta = Math.min(alpha * sMnorm, sigma2 * delta);
            else if (actred < eta1 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * sMnorm, sigma2 * delta));
            else if (actred < eta2 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * sMnorm, sigma3 * delta));
            else {
                if (reach_boundary.get()) {
                    delta = sigma3 * delta;
                } else {
                    delta = Math.max(delta, Math.min(alpha * sMnorm, sigma3 * delta));
                }
            }

            info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d%n", iter, actred, prered, delta, f, gnorm, cg_iter);

            if (actred > eta0 * prered) {
                iter++;
                System.arraycopy(w_new, 0, w, 0, n);
                f = fnew;
                fun_obj.grad(w, g);

                gnorm = euclideanNorm(g);
                if (gnorm <= eps * gnorm0)
                    break;
            }
            if (f < -1.0e+32) {
                info("WARNING: f < -1.0e+32%n");
                break;
            }
            if (prered <= 0) {
                info("WARNING: prered <= 0%n");
                break;
            }
            if (Math.abs(actred) <= 1.0e-12 * Math.abs(f) && Math.abs(prered) <= 1.0e-12 * Math.abs(f)) {
                info("WARNING: actred and prered too small%n");
                break;
            }
        }
    }

    private int trpcg(double delta, double[] g, double[] M, double[] s, double[] r, AtomicBoolean reach_boundary) {
        int n = fun_obj.get_nr_variable();
        double one = 1;
        double[] d = new double[n];
        double[] Hd = new double[n];
        double zTr, znewTrnew, alpha, beta, cgtol;
        double[] z = new double[n];

        reach_boundary.set(false);
        for (int i = 0; i < n; i++) {
            s[i] = 0;
            r[i] = -g[i];
            z[i] = r[i] / M[i];
            d[i] = z[i];
        }

        zTr = dot(z, r);
        cgtol = eps_cg * Math.sqrt(zTr);
        int cg_iter = 0;

        while (true) {
            if (Math.sqrt(zTr) <= cgtol)
                break;
            cg_iter++;
            fun_obj.Hv(d, Hd);

            alpha = zTr / dot(d, Hd);
            daxpy(alpha, d, s);

            double sMnorm = Math.sqrt(uTMv(n, s, M, s));
            if (sMnorm > delta) {
                info("cg reaches trust region boundary%n");
                reach_boundary.set(true);
                alpha = -alpha;
                daxpy(alpha, d, s);

                double sTMd = uTMv(n, s, M, d);
                double sTMs = uTMv(n, s, M, s);
                double dTMd = uTMv(n, d, M, d);
                double dsq = delta * delta;
                double rad = Math.sqrt(sTMd * sTMd + dTMd * (dsq - sTMs));
                if (sTMd >= 0)
                    alpha = (dsq - sTMs) / (sTMd + rad);
                else
                    alpha = (rad - sTMd) / dTMd;
                daxpy(alpha, d, s);
                alpha = -alpha;
                daxpy(alpha, Hd, r);
                break;
            }
            alpha = -alpha;
            daxpy(alpha, Hd, r);

            for (int i = 0; i < n; i++)
                z[i] = r[i] / M[i];
            znewTrnew = dot(z, r);
            beta = znewTrnew / zTr;
            scale(beta, d);
            daxpy(one, z, d);
            zTr = znewTrnew;
        }

        return (cg_iter);
    }

    /**
     * constant times a vector plus a vector
     *
     * <pre>
     * vector2 += constant * vector1
     * </pre>
     *
     * @since 1.8
     */
    private static void daxpy(double constant, double vector1[], double vector2[]) {
        if (constant == 0) return;

        assert vector1.length == vector2.length;
        for (int i = 0; i < vector1.length; i++) {
            vector2[i] += constant * vector1[i];
        }
    }

    /**
     * returns the dot product of two vectors
     *
     * @since 1.8
     */
    private static double dot(double vector1[], double vector2[]) {

        double product = 0;
        assert vector1.length == vector2.length;
        for (int i = 0; i < vector1.length; i++) {
            product += vector1[i] * vector2[i];
        }
        return product;

    }

    /**
     * returns the euclidean norm of a vector
     *
     * @since 1.8
     */
    private static double euclideanNorm(double vector[]) {

        int n = vector.length;

        if (n < 1) {
            return 0;
        }

        if (n == 1) {
            return Math.abs(vector[0]);
        }

        // this algorithm is (often) more accurate than just summing up the squares and taking the square-root afterwards

        double scale = 0; // scaling factor that is factored out
        double sum = 1; // basic sum of squares from which scale has been factored out
        for (int i = 0; i < n; i++) {
            if (vector[i] != 0) {
                double abs = Math.abs(vector[i]);
                // try to get the best scaling factor
                if (scale < abs) {
                    double t = scale / abs;
                    sum = 1 + sum * (t * t);
                    scale = abs;
                } else {
                    double t = abs / scale;
                    sum += t * t;
                }
            }
        }

        return scale * Math.sqrt(sum);
    }

    /**
     * scales a vector by a constant
     *
     * @since 1.8
     */
    private static void scale(double constant, double vector[]) {
        if (constant == 1.0) return;
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= constant;
        }
    }

    private static double uTMv(int n, double[] u, double[] M, double[] v) {
        int m = n - 4;
        double res = 0;
        int i;
        for (i = 0; i < m; i += 5)
            res += u[i] * M[i] * v[i] + u[i + 1] * M[i + 1] * v[i + 1] + u[i + 2] * M[i + 2] * v[i + 2] +
                    u[i + 3] * M[i + 3] * v[i + 3] + u[i + 4] * M[i + 4] * v[i + 4];
        for (; i < n; i++)
            res += u[i] * M[i] * v[i];
        return res;
    }

}
