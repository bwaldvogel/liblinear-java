package de.bwaldvogel.liblinear;

class MutableInt {

    private boolean initialized;
    private int     value;

    public MutableInt(int value) {
        set(value);
    }

    void set(int value) {
        this.value = value;
        this.initialized = true;
    }

    int get() {
        if (!initialized) {
            throw new IllegalStateException("Value not yet initialized");
        }
        return value;
    }

}
