package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.SolverType.*;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Serializable;
import java.io.Writer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;


/**
 * <p>Model stores the model obtained from the training procedure</p>
 *
 * <p>use {@link Linear#loadModel(Path)} and {@link Linear#saveModel(Path, Model)} to load/save it</p>
 */
public final class Model implements Serializable {

    private static final long serialVersionUID = -6456047576741854834L;

    double bias;

    /** label of each class */
    int[] label;

    int nr_class;

    int nr_feature;

    SolverType solverType;

    /** feature weight array */
    double[] w;

    /** one-class SVM only */
    double rho;

    /**
     * @return number of classes
     */
    public int getNrClass() {
        return nr_class;
    }

    /**
     * @return number of features
     */
    public int getNrFeature() {
        return nr_feature;
    }

    public int[] getLabels() {
        return Arrays.copyOf(label, nr_class);
    }

    public SolverType getSolverType() {
        return solverType;
    }

    /**
     * The array w gives feature weights; its size is
     * nr_feature*nr_class but is nr_feature if nr_class = 2. We use one
     * against the rest for multi-class classification, so each feature
     * index corresponds to nr_class weight values. Weights are
     * organized in the following way
     *
     * <pre>
     * +------------------+------------------+------------+
     * | nr_class weights | nr_class weights |  ...
     * | for 1st feature  | for 2nd feature  |
     * +------------------+------------------+------------+
     * </pre>
     *
     * If bias &gt;= 0, x becomes [x; bias]. The number of features is
     * increased by one, so w is a (nr_feature+1)*nr_class array. The
     * value of bias is stored in the variable bias.
     * @see #getBias()
     * @return a <b>copy of</b> the feature weight array as described
     */
    public double[] getFeatureWeights() {
        return Arrays.copyOf(w, w.length);
    }

    /**
     * @return true for logistic regression solvers
     */
    public boolean isProbabilityModel() {
        return solverType.isLogisticRegressionSolver();
    }

    /**
     * @see #getFeatureWeights()
     */
    public double getBias() {
        return bias;
    }

    private double get_w_value(int idx, int label_idx) {
        if (idx < 0 || idx > nr_feature) {
            return 0;
        }
        if (solverType.isSupportVectorRegression() || solverType.isOneClass()) {
            return w[idx];
        } else {
            if (label_idx < 0 || label_idx >= nr_class) {
                return 0;
            }
            if (nr_class == 2 && solverType != MCSVM_CS) {
                if (label_idx == 0) {
                    return w[idx];
                } else {
                    return -w[idx];
                }
            } else {
                return w[idx * nr_class + label_idx];
            }
        }
    }

    /**
     * This function gives the coefficient for the feature with feature index =
     * feat_idx and the class with label index = label_idx. Note that feat_idx
     * starts from 1, while label_idx starts from 0. If feat_idx is not in the
     * valid range (1 to nr_feature), then a zero value will be returned. For
     * classification models, if label_idx is not in the valid range (0 to
     * nr_class-1), then a zero value will be returned; for regression models,
     * label_idx is ignored.
     *
     * @since 1.95
     */
    // feat_idx: starting from 1 to nr_feature
    // label_idx: starting from 0 to nr_class-1 for classification models;
    //            for regression and one-class SVM models, label_idx is
    //            ignored.
    public double getDecfunCoef(int featIdx, int labelIdx) {
        if (featIdx > nr_feature) {
            return 0;
        }
        return get_w_value(featIdx - 1, labelIdx);
    }

    /**
     * This function gives the bias term corresponding to the class with the
     * label_idx. For classification models, if label_idx is not in a valid range
     * (0 to nr_class-1), then a zero value will be returned; for regression
     * models, label_idx is ignored.
     *
     * @since 1.95
     */
    public double getDecfunBias(int labelIdx) {
        if (solverType.isOneClass()) {
            throw new IllegalArgumentException("Can not be called for a one-class SVM model");
        }
        int biasIdx = nr_feature;
        if (bias <= 0) {
            return 0;
        } else {
            return bias * get_w_value(biasIdx, labelIdx);
        }
    }

    /**
     * This function gives rho, the bias term used in one-class SVM only.
     *
     * This function can only be called for a one-class SVM model.
     *
     * @since 2.40
     */
    public double getDecfunRho() {
        if (!solverType.isOneClass()) {
            throw new IllegalArgumentException("Can be called only for a one-class SVM model");
        }
        return rho;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Model");
        sb.append(" bias=").append(bias);
        sb.append(" nr_class=").append(nr_class);
        sb.append(" nr_feature=").append(nr_feature);
        sb.append(" solverType=").append(solverType);
        return sb.toString();
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(getBias(), nr_class, nr_feature, getSolverType(), rho);
        result = 31 * result + Arrays.hashCode(label);
        result = 31 * result + arrayHashCode(w);
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Model model = (Model)o;
        return Double.compare(model.getBias(), getBias()) == 0
            && nr_class == model.nr_class
            && nr_feature == model.nr_feature
            && Double.compare(model.rho, rho) == 0
            && Arrays.equals(label, model.label)
            && getSolverType() == model.getSolverType()
            && arrayEquals(w, model.w);
    }

    /**
     * don't use {@link Arrays#equals(double[], double[])} here, cause 0.0 and -0.0 should be handled the same
     *
     * @see Linear#saveModel(java.io.Writer, Model)
     */
    private static boolean arrayEquals(double[] a, double[] a2) {
        if (a == a2)
            return true;
        if (a == null || a2 == null)
            return false;

        int length = a.length;
        if (a2.length != length)
            return false;

        for (int i = 0; i < length; i++)
            if (a[i] != a2[i])
                return false;

        return true;
    }

    /**
     * see {@link Arrays#hashCode(double[])} but treat 0.0 and -0.0 the same
     */
    private static int arrayHashCode(double[] w) {
        if (w == null)
            return 0;

        int result = 1;
        for (double element : w) {
            if (element == -0.0) {
                element = 0.0;
            }
            long bits = Double.doubleToLongBits(element);
            result = 31 * result + (int)(bits ^ (bits >>> 32));
        }
        return result;
    }

    /**
     * @deprecated use {@link Model#save(Path)} instead
     */
    public void save(File modelFile) throws IOException {
        save(modelFile.toPath());
    }

    /**
     * see {@link Linear#saveModel(Path, Model)}
     */
    public void save(Path modelPath) throws IOException {
        Linear.saveModel(modelPath, this);
    }

    /**
     * see {@link Linear#saveModel(Writer, Model)}
     */
    public void save(Writer writer) throws IOException {
        Linear.saveModel(writer, this);
    }

    /**
     * @deprecated use {@link Model#load(Path)} instead
     */
    public static Model load(File modelFile) throws IOException {
        return load(modelFile.toPath());
    }

    /**
     * see {@link Linear#loadModel(Path)}
     */
    public static Model load(Path modelPath) throws IOException {
        return Linear.loadModel(modelPath);
    }

    /**
     * see {@link Linear#loadModel(Reader)}
     */
    public static Model load(Reader inputReader) throws IOException {
        return Linear.loadModel(inputReader);
    }
}
