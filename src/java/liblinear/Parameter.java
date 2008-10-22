package liblinear;

import java.util.Arrays;


/**
 *  C is the cost of constraints violation. (we usually use 1 to 1000)
 *  eps is the stopping criterion. (we usually use 0.01).
 *
 *  nr_weight, weight_label, and weight are used to change the penalty
 *  for some classes (If the weight for a class is not changed, it is
 *  set to 1). This is useful for training classifier using unbalanced
 *  input data or with asymmetric misclassification cost.
 *
 *  nr_weight is the number of elements in the array weight_label and
 *  weight. Each weight[i] corresponds to weight_label[i], meaning that
 *  the penalty of class weight_label[i] is scaled by a factor of weight[i].
 *
 *  If you do not want to change penalty for any of the classes,
 *  just set nr_weight to 0.
 *
 *  *NOTE* To avoid wrong parameters, check_parameter() should be
 *  called before train().
 */
public final class Parameter {

   double     C;

   /** stopping criteria */
   double     eps;

   SolverType solverType;

   double[]   weight      = null;

   int[]      weightLabel = null;

   /* these are for training only */

   public Parameter( SolverType solverType, double C, double eps ) {
      setSolverType(solverType);
      setC(C);
      setEps(eps);
   }

   public void setWeights( double[] weights, int[] weightLabels ) {
      if ( weights == null ) throw new IllegalArgumentException("weight must not be null");
      if ( weightLabels == null || weightLabels.length != weights.length ) throw new IllegalArgumentException("weightlabels must have same length as weight");
      this.weightLabel = Arrays.copyOf(weightLabels, weightLabels.length);
      this.weight = Arrays.copyOf(weights, weights.length);
   }

   public int getNumWeights() {
      if ( weight == null ) return 0;
      return weight.length;
   }

   public void setC( double C ) {
      if ( C <= 0 ) throw new IllegalArgumentException("C must not be <= 0");
      this.C = C;
   }

   public void setEps( double eps ) {
      if ( eps <= 0 ) throw new IllegalArgumentException("eps must not be <= 0");
      this.eps = eps;
   }

   public void setSolverType( SolverType solverType ) {
      if ( solverType == null ) throw new IllegalArgumentException("solver type must not be null");
      this.solverType = solverType;
   }
}
