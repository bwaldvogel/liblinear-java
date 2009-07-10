package liblinear;

import java.io.Serializable;
import java.util.Arrays;

import static liblinear.Linear.copyOf;


/**
 * use {@link Linear#loadModel(String)} and {@link Linear#saveModel(String, Model)} to load/save it
 */
public final class Model implements Serializable {

   private static final long serialVersionUID = -6456047576741854834L;

   double                    bias;

   /** label of each class (label[n]) */
   int[]                     label;

   /** number of classes */
   int                       nr_class;

   int                       nr_feature;

   SolverType                solverType;

   double[]                  w;

   public int getNrClass() {
      return nr_class;
   }

   public int getNrFeature() {
      return nr_feature;
   }

   public int[] getLabels() {
      return copyOf(label, nr_class);
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
      final int prime = 31;
      int result = 1;
      long temp;
      temp = Double.doubleToLongBits(bias);
      result = prime * result + (int)(temp ^ (temp >>> 32));
      result = prime * result + Arrays.hashCode(label);
      result = prime * result + nr_class;
      result = prime * result + nr_feature;
      result = prime * result + ((solverType == null) ? 0 : solverType.hashCode());
      result = prime * result + Arrays.hashCode(w);
      return result;
   }

   @Override
   public boolean equals( Object obj ) {
      if ( this == obj ) return true;
      if ( obj == null ) return false;
      if ( getClass() != obj.getClass() ) return false;
      Model other = (Model)obj;
      if ( Double.doubleToLongBits(bias) != Double.doubleToLongBits(other.bias) ) return false;
      if ( !Arrays.equals(label, other.label) ) return false;
      if ( nr_class != other.nr_class ) return false;
      if ( nr_feature != other.nr_feature ) return false;
      if ( solverType == null ) {
         if ( other.solverType != null ) return false;
      } else if ( !solverType.equals(other.solverType) ) return false;
      if ( !equals(w, other.w) ) return false;
      return true;
   }

   /**
    * don't use {@link Arrays#equals(double[], double[])} here, cause 0.0 and -0.0 should be handled the same
    *
    * @see Linear#saveModel(java.io.Writer, Model)
    */
   protected static boolean equals( double[] a, double[] a2 ) {
      if ( a == a2 ) return true;
      if ( a == null || a2 == null ) return false;

      int length = a.length;
      if ( a2.length != length ) return false;

      for ( int i = 0; i < length; i++ )
         if ( a[i] != a2[i] ) return false;

      return true;
   }
}
