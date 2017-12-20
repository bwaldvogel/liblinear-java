package de.bwaldvogel.liblinear;

import java.util.Arrays;

class ThreadLocalDoubleArray extends ThreadLocal<double[]> {
    private final int length;

    public ThreadLocalDoubleArray(int length) {
        this.length = length;
    }

    @Override
    public double[] get() {
        double[] res = super.get();
        Arrays.fill(res, 0);
        return res;
    }

    @Override
    protected double[] initialValue() {
        return new double[this.length];
    }
}
