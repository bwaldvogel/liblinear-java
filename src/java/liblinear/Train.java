package liblinear;

import static liblinear.Linear.NL;
import static liblinear.Linear.atof;
import static liblinear.Linear.atoi;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;


public class Train {

   public static void main( String[] args ) throws IOException, InvalidInputDataException {
      new Train().run(args);
   }

   private double    bias             = 1;
   private boolean   cross_validation = false;
   private String    inputFilename;
   private String    modelFilename;
   private int       nr_fold;
   private Parameter param            = null;
   private Problem   prob             = null;

   private void do_cross_validation() {
      int[] target = new int[prob.l];

      long start, stop;
      start = System.currentTimeMillis();
      Linear.crossValidation(prob, param, nr_fold, target);
      stop = System.currentTimeMillis();
      System.out.println("time: " + (stop - start) + " ms");

      int total_correct = 0;
      for ( int i = 0; i < prob.l; i++ )
         if ( target[i] == prob.y[i] ) ++total_correct;

      System.out.printf("correct: %d" + NL, total_correct);
      System.out.printf("Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob.l);
   }

   private void exit_with_help() {
      System.out.println("Usage: train [options] training_set_file [model_file]" + NL //
         + "options:" + NL//
         + "-s type : set type of solver (default 1)" + NL//
         + "   0 -- L2-regularized logistic regression" + NL//
         + "   1 -- L2-loss support vector machines (dual)" + NL//
         + "   2 -- L2-loss support vector machines (primal)" + NL//
         + "   3 -- L1-loss support vector machines (dual)" + NL//
         + "   4 -- multi-class support vector machines by Crammer and Singer" + NL//
         + "-c cost : set the parameter C (default 1)" + NL//
         + "-e epsilon : set tolerance of termination criterion" + NL//
         + "   -s 0 and 2" + NL//
         + "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2," + NL//
         + "       where f is the primal function, (default 0.01)" + NL//
         + "   -s 1, 3, and 4" + NL//
         + "       Dual maximal violation <= eps; similar to libsvm (default 0.1)" + NL//
         + "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default 1)" + NL//
         + "-wi weight: weights adjust the parameter C of different classes (see README for details)" + NL//
         + "-v n: n-fold cross validation mode" + NL//
      );
      System.exit(1);
   }


   Problem getProblem() {
      return prob;
   }

   double getBias() {
      return bias;
   }

   Parameter getParameter() {
      return param;
   }

   void parse_command_line( String argv[] ) {
      int i;

      // eps: see setting below
      param = new Parameter(SolverType.L2LOSS_SVM_DUAL, 1, Double.POSITIVE_INFINITY);
      // default values
      bias = 1;
      cross_validation = false;

      int nr_weight = 0;

      // parse options
      for ( i = 0; i < argv.length; i++ ) {
         if ( argv[i].charAt(0) != '-' ) break;
         if ( ++i >= argv.length ) exit_with_help();
         switch ( argv[i - 1].charAt(1) ) {
         case 's':
            param.solverType = SolverType.values()[atoi(argv[i])];
            break;
         case 'c':
            param.setC(atof(argv[i]));
            break;
         case 'e':
            param.setEps(atof(argv[i]));
            break;
         case 'B':
            bias = atof(argv[i]);
            break;
         case 'w':
            ++nr_weight;
            {
               int[] old = param.weightLabel;
               param.weightLabel = new int[nr_weight];
               System.arraycopy(old, 0, param.weightLabel, 0, nr_weight - 1);
            }

            {
               double[] old = param.weight;
               param.weight = new double[nr_weight];
               System.arraycopy(old, 0, param.weight, 0, nr_weight - 1);
            }

            param.weightLabel[nr_weight - 1] = atoi(argv[i - 1].substring(2));
            param.weight[nr_weight - 1] = atof(argv[i]);
            break;
         case 'v':
            cross_validation = true;
            nr_fold = atoi(argv[i]);
            if ( nr_fold < 2 ) {
               System.err.print("n-fold cross validation: n must >= 2\n");
               exit_with_help();
            }
            break;
         default:
            System.err.println("unknown option");
            exit_with_help();
         }
      }

      // determine filenames

      if ( i >= argv.length ) exit_with_help();

      inputFilename = argv[i];

      if ( i < argv.length - 1 )
         modelFilename = argv[i + 1];
      else {
         int p = argv[i].lastIndexOf('/');
         ++p; // whew...
         modelFilename = argv[i].substring(p) + ".model";
      }

      if ( param.eps == Double.POSITIVE_INFINITY ) {
         if ( param.solverType == SolverType.L2_LR || param.solverType == SolverType.L2LOSS_SVM ) {
            param.setEps(0.01);
         } else if ( param.solverType == SolverType.L2LOSS_SVM_DUAL || param.solverType == SolverType.L1LOSS_SVM_DUAL
            || param.solverType == SolverType.MCSVM_CS ) {
            param.setEps(0.1);
         }
      }
   }

   /**
    * reads a problem from LibSVM format
    * @param filename the name of the svm file
    * @throws IOException obviously in case of any I/O exception ;)
    * @throws InvalidInputDataException if the input file is not correctly formatted
    */
   void readProblem( String filename ) throws IOException, InvalidInputDataException {
      BufferedReader fp = new BufferedReader(new FileReader(filename));
      List<Integer> vy = new ArrayList<Integer>();
      List<FeatureNode[]> vx = new ArrayList<FeatureNode[]>();
      int max_index = 0;

      int lineNr = 0;

      try {
         while ( true ) {
            String line = fp.readLine();
            if ( line == null ) break;
            lineNr++;

            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
            String token = st.nextToken();

            try {
               vy.add(atoi(token));
            }
            catch ( NumberFormatException e ) {
               throw new InvalidInputDataException("invalid label: " + token, filename, lineNr, e);
            }

            int m = st.countTokens() / 2;
            FeatureNode[] x;
            if ( bias >= 0 ) {
               x = new FeatureNode[m + 1];
            } else {
               x = new FeatureNode[m];
            }
            int indexBefore = 0;
            for ( int j = 0; j < m; j++ ) {

               token = st.nextToken();
               int index;
               try {
                  index = atoi(token);
               }
               catch ( NumberFormatException e ) {
                  throw new InvalidInputDataException("invalid index: " + token, filename, lineNr, e);
               }

               // assert that indices are valid and sorted
               if ( index < 0 ) throw new InvalidInputDataException("invalid index: " + index, filename, lineNr);
               if ( index <= indexBefore ) throw new InvalidInputDataException("indices must be sorted in ascending order", filename, lineNr);
               indexBefore = index;

               token = st.nextToken();
               try {
                  double value = atof(token);
                  x[j] = new FeatureNode(index, value);
               }
               catch ( NumberFormatException e ) {
                  throw new InvalidInputDataException("invalid value: " + token, filename, lineNr);
               }
            }
            if ( m > 0 ) {
               max_index = Math.max(max_index, x[m - 1].index);
            }

            vx.add(x);
         }

         prob = constructProblem(vy, vx, max_index);
      }
      finally {
         fp.close();
      }
   }

   private Problem constructProblem( List<Integer> vy, List<FeatureNode[]> vx, int max_index ) {
      Problem prob = new Problem();
      prob.bias = bias;
      prob.l = vy.size();
      prob.n = max_index;
      if ( bias >= 0 ) {
         prob.n++;
      }
      prob.x = new FeatureNode[prob.l][];
      for ( int i = 0; i < prob.l; i++ ) {
         prob.x[i] = vx.get(i);

         if ( bias >= 0 ) {
            assert prob.x[i][prob.x[i].length - 1] == null;
            prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
         } else {
            assert prob.x[i][prob.x[i].length - 1] != null;
         }
      }

      prob.y = new int[prob.l];
      for ( int i = 0; i < prob.l; i++ )
         prob.y[i] = vy.get(i);

      return prob;
   }

   private void run( String[] args ) throws IOException, InvalidInputDataException {
      parse_command_line(args);
      readProblem(inputFilename);
      if ( cross_validation )
         do_cross_validation();
      else {
         Model model = Linear.train(prob, param);
         Linear.saveModel(new File(modelFilename), model);
      }
   }
}
