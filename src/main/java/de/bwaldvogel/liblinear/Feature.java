package de.bwaldvogel.liblinear;


public interface Feature {

    int getIndex();

    double getValue();

    void setValue(double value);
}
