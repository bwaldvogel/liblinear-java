package liblinear;

import static org.easymock.EasyMock.anyInt;
import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.EasyMock.isA;
import static org.easymock.classextension.EasyMock.createMock;
import static org.easymock.classextension.EasyMock.replay;
import static org.easymock.classextension.EasyMock.verify;
import static org.fest.assertions.Assertions.assertThat;
import static org.fest.assertions.Fail.fail;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
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

   private Model createSomeModel() {
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

      OutputStream out = createMock(OutputStream.class);
      Object[] mocks = new Object[] { out };

      IOException ioException = new IOException("some reason");

      out.write(isA(byte[].class), anyInt(), anyInt());
      expectLastCall().anyTimes();
      out.write(anyInt());
      expectLastCall().anyTimes();
      out.write(isA(byte[].class));
      expectLastCall().anyTimes();
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
