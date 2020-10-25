package de.bwaldvogel.liblinear;

interface Function {

    double fun(double[] w);

    void grad(double[] w, double[] g);

    void Hv(double[] s, double[] Hs);

    int get_nr_variable();

    void get_diag_preconditioner(double[] M);

    // Note: This implementation is unused but function::linesearch_and_update
    // from upstream newton.cpp seems to be unused as well
    default double linesearch_and_update(double[] w, double[] s, MutableDouble f, double[] g, double alpha) {
        double gTs = 0;
        double eta = 0.01;
        int n = get_nr_variable();
        int max_num_linesearch = 20;
        double[] w_new = new double[n];
        double fold = f.get();

        for (int i = 0; i < n; i++)
            gTs += s[i] * g[i];

        int num_linesearch = 0;
        for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
            for (int i = 0; i < n; i++)
                w_new[i] = w[i] + alpha * s[i];
            f.set(fun(w_new));
            if (f.get() - fold <= eta * alpha * gTs)
                break;
            else
                alpha *= 0.5;
        }

        if (num_linesearch >= max_num_linesearch) {
            f.set(fold);
            return 0;
        } else
            System.arraycopy(w_new, 0, w, 0, n);

        return alpha;
    }

}
