package linear;

/**
 *  Problem describes the problem
 *
 *  <pre>
 *  For example, if we have the following training data:
 *
 *  LABEL       ATTR1   ATTR2   ATTR3   ATTR4   ATTR5
 *  -----       -----   -----   -----   -----   -----
 *  1           0       0.1     0.2     0       0
 *  2           0       0.1     0.3    -1.2     0
 *  1           0.4     0       0       0       0
 *  2           0       0.1     0       1.4     0.5
 *  3          -0.1    -0.2     0.1     1.1     0.1
 *
 *  and bias = 1, then the components of problem are:
 *
 *  l = 5
 *  n = 6
 *
 *  y -> 1 2 1 2 3
 *
 *  x -> [ ] -> (2,0.1) (3,0.2) (6,1) (-1,?)
 *       [ ] -> (2,0.1) (3,0.3) (4,-1.2) (6,1) (-1,?)
 *       [ ] -> (1,0.4) (6,1) (-1,?)
 *       [ ] -> (2,0.1) (4,1.4) (5,0.5) (6,1) (-1,?)
 *       [ ] -> (1,-0.1) (2,-0.2) (3,0.1) (4,1.1) (5,0.1) (6,1) (-1,?)
 * </pre>
 */
public class Problem {

   /** the number of training data */
   public int             l;

   /** the number of feature (including the bias feature if bias >= 0) */
   public int             n;

   /** an array containing the target values */
   public int[]           y;

   /** array of sparse feature nodes */
   public FeatureNode[][] x;

   /**
    * If bias >= 0, we assume that one additional feature is added
    * to the end of each data instance
    */
   public double          bias;
}
