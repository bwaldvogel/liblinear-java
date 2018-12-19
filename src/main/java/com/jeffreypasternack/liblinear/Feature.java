package com.jeffreypasternack.liblinear;

/**
 * @since 1.9
 */
public interface Feature {

    int getIndex();

    double getValue();

    void setValue(double value);
}
