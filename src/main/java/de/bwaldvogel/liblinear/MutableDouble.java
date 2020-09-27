package de.bwaldvogel.liblinear;

class MutableDouble {

    private boolean initialized;
    private double  value;

    void set(double value) {
        this.value = value;
        this.initialized = true;
    }

    double get() {
        if (!initialized) {
            throw new IllegalStateException("Value not yet initialized");
        }
        return value;
    }

}
