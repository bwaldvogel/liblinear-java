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
import java.util.Arrays;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Test;


public class LinearTest {

   private static Random random = new Random();

   @BeforeClass
   public static void disableDebugOutput() {
      Linear.disableDebugOutput();
   }

   @Test
   public void testRealloc() {

      int[] f = new int[] { 1, 2, 3 };
      f = Arrays.copyOf(f, 5);
      f[3] = 4;
      f[4] = 5;
      assertThat(f).isEqualTo(new int[] { 1, 2, 3, 4, 5 });
   }

   public static Model createSomeModel() {
      Model model = new Model();
      model.solverType = SolverType.L2_LR;
      model.bias = 2;
      model.label = new int[] { 1, Integer.MAX_VALUE, 2 };
      model.w = new double[model.label.length * 300];
      for ( int i = 0; i < model.w.length; i++ ) {
         // precision should be at least 1e-4
         model.w[i] = Math.round(random.nextDouble() * 100000.0) / 10000.0;
      }
      model.nr_feature = model.w.length / model.label.length - 1;
      model.nr_class = model.label.length;
      return model;
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
         model = createSomeModel();
         model.solverType = solverType;

         File tempFile = File.createTempFile("liblinear", "modeltest");
         tempFile.deleteOnExit();
         Linear.saveModel(tempFile, model);

         Model loadedModel = Linear.loadModel(tempFile);
         assertThat(loadedModel).isEqualTo(model);
      }
   }

   @Test
   public void testSaveModelWithIOException() throws Exception {
      Model model = createSomeModel();

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
}
