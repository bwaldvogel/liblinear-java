package liblinear;

import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.classextension.EasyMock.createNiceMock;
import static org.easymock.classextension.EasyMock.replay;
import static org.easymock.classextension.EasyMock.verify;
import static org.fest.assertions.Assertions.assertThat;
import static org.fest.assertions.Fail.fail;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.junit.BeforeClass;
import org.junit.Test;


public class LinearTest {

   private static Random random = new Random(0);

   @BeforeClass
   public static void disableDebugOutput() {
      Linear.disableDebugOutput();
   }

   public static Model createRandomModel() {
      Model model = new Model();
      model.solverType = SolverType.L2_LR;
      model.bias = 2;
      model.label = new int[] { 1, Integer.MAX_VALUE, 2 };
      model.w = new double[model.label.length * 300];
      for ( int i = 0; i < model.w.length; i++ ) {
         // precision should be at least 1e-4
         model.w[i] = Math.round(random.nextDouble() * 100000.0) / 10000.0;
      }

      // force at least one value to be zero
      model.w[random.nextInt(model.w.length)] = 0.0;
      model.w[random.nextInt(model.w.length)] = -0.0;

      model.nr_feature = model.w.length / model.label.length - 1;
      model.nr_class = model.label.length;
      return model;
   }

   public static Problem createRandomProblem( int numClasses ) {
      Problem prob = new Problem();
      prob.bias = -1;
      prob.l = random.nextInt(100) + 1;
      prob.n = random.nextInt(100) + 1;
      prob.x = new FeatureNode[prob.l][];
      prob.y = new int[prob.l];

      for ( int i = 0; i < prob.l; i++ ) {

         prob.y[i] = random.nextInt(numClasses);

         Set<Integer> randomNumbers = new TreeSet<Integer>();
         int num = random.nextInt(prob.n) + 1;
         for ( int j = 0; j < num; j++ ) {
            randomNumbers.add(random.nextInt(prob.n) + 1);
         }
         List<Integer> randomIndices = new ArrayList<Integer>(randomNumbers);
         Collections.sort(randomIndices);

         prob.x[i] = new FeatureNode[randomIndices.size()];
         for ( int j = 0; j < randomIndices.size(); j++ ) {
            prob.x[i][j] = new FeatureNode(randomIndices.get(j), random.nextDouble());
         }
      }
      return prob;
   }

   @Test
   public void testRealloc() {

      int[] f = new int[] { 1, 2, 3 };
      f = Linear.copyOf(f, 5);
      f[3] = 4;
      f[4] = 5;
      assertThat(f).isEqualTo(new int[] { 1, 2, 3, 4, 5 });
   }

   @Test
   public void testAtoi() {
      assertThat(Linear.atoi("+25")).isEqualTo(25);
      assertThat(Linear.atoi("-345345")).isEqualTo(-345345);
      assertThat(Linear.atoi("+0")).isEqualTo(0);
      assertThat(Linear.atoi("0")).isEqualTo(0);
      assertThat(Linear.atoi("2147483647")).isEqualTo(Integer.MAX_VALUE);
      assertThat(Linear.atoi("-2147483648")).isEqualTo(Integer.MIN_VALUE);
   }

   @Test(expected = NumberFormatException.class)
   public void testAtoiInvalidData() {
      Linear.atoi("+");
   }

   @Test(expected = NumberFormatException.class)
   public void testAtoiInvalidData2() {
      Linear.atoi("abc");
   }

   @Test(expected = NumberFormatException.class)
   public void testAtoiInvalidData3() {
      Linear.atoi(" ");
   }

   @Test
   public void testAtof() {
      assertThat(Linear.atof("+25")).isEqualTo(25);
      assertThat(Linear.atof("-25.12345678")).isEqualTo(-25.12345678);
      assertThat(Linear.atof("0.345345299")).isEqualTo(0.345345299);
   }

   @Test(expected = NumberFormatException.class)
   public void testAtofInvalidData() {
      Linear.atof("0.5t");
   }

   @Test
   public void testLoadSaveModel() throws Exception {

      Model model = null;
      for ( SolverType solverType : SolverType.values() ) {
         model = createRandomModel();
         model.solverType = solverType;

         File tempFile = File.createTempFile("liblinear", "modeltest");
         tempFile.deleteOnExit();
         Linear.saveModel(tempFile, model);

         Model loadedModel = Linear.loadModel(tempFile);
         assertThat(loadedModel).isEqualTo(model);
      }
   }

   @Test
   public void testCrossValidation() throws Exception {

      int numClasses = random.nextInt(10) + 1;

      Problem prob = createRandomProblem(numClasses);

      Parameter param = new Parameter(SolverType.L2_LR, 10, 0.01);
      int nr_fold = 10;
      int[] target = new int[prob.l];
      Linear.crossValidation(prob, param, nr_fold, target);

      for ( int clazz : target ) {
         assertThat(clazz).isGreaterThanOrEqualTo(0).isLessThan(numClasses);
      }
   }

   @Test
   public void testSaveModelWithIOException() throws Exception {
      Model model = createRandomModel();

      Writer out = createNiceMock(Writer.class);
      Object[] mocks = new Object[] { out };

      IOException ioException = new IOException("some reason");

      out.flush();
      expectLastCall().andThrow(ioException);
      out.close();
      expectLastCall().times(1);

      replay(mocks);
      try {
         Linear.saveModel(out, model);
         fail("IOException expected");
      }
      catch ( IOException e ) {
         assertThat(e).isEqualTo(ioException);
      }
      verify(mocks);
   }

   /**
    * create a very simple problem and check if the clearly separated examples are recognized as such
    */
   @Test
   public void testTrain() {
      Problem prob = new Problem();
      prob.bias = -1;
      prob.l = 4;
      prob.n = 4;
      prob.x = new FeatureNode[4][];
      prob.x[0] = new FeatureNode[2];
      prob.x[1] = new FeatureNode[1];
      prob.x[2] = new FeatureNode[1];
      prob.x[3] = new FeatureNode[3];

      prob.x[0][0] = new FeatureNode(1, 1);
      prob.x[0][1] = new FeatureNode(2, 1);

      prob.x[1][0] = new FeatureNode(3, 1);
      prob.x[2][0] = new FeatureNode(3, 1);

      prob.x[3][0] = new FeatureNode(1, 2);
      prob.x[3][1] = new FeatureNode(2, 1);
      prob.x[3][2] = new FeatureNode(4, 1);

      prob.y = new int[4];
      prob.y[0] = 0;
      prob.y[1] = 1;
      prob.y[2] = 1;
      prob.y[3] = 0;

      for ( SolverType solver : SolverType.values() ) {
         for ( double C = 0.1; C <= 100.; C *= 10. ) {
            Parameter param = new Parameter(solver, C, 0.1);
            Model model = Linear.train(prob, param);

            int i = 0;
            for ( int value : prob.y ) {
               int prediction = Linear.predict(model, prob.x[i]);
               assertThat(prediction).isEqualTo(value);
               i++;
            }
         }
      }
   }

   @Test(expected = IllegalArgumentException.class)
   public void testTrainUnsortedProblem() {
      Problem prob = new Problem();
      prob.bias = -1;
      prob.l = 1;
      prob.n = 2;
      prob.x = new FeatureNode[4][];
      prob.x[0] = new FeatureNode[2];

      prob.x[0][0] = new FeatureNode(2, 1);
      prob.x[0][1] = new FeatureNode(1, 1);

      prob.y = new int[4];
      prob.y[0] = 0;

      Parameter param = new Parameter(SolverType.L2_LR, 10, 0.1);
      try {
         Linear.train(prob, param);
      }
      catch ( IllegalArgumentException e ) {
         assertThat(e).message().contains("nodes").contains("sorted").contains("ascending").contains("order");
         throw e;
      }
   }
}
