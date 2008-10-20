package linear;

import static org.fest.assertions.Assertions.assertThat;

import java.io.File;
import java.util.Arrays;

import org.junit.BeforeClass;
import org.junit.Test;


public class LinearTest {

   @BeforeClass
   public static void disableDebugOutput() {
      Linear.DEBUG_OUTPUT = false;
   }

   @Test
   public void testRealloc() {

      int[] f = new int[] { 1, 2, 3 };
      f = Arrays.copyOf(f, 5);
      f[3] = 4;
      f[4] = 5;
      assertThat(f).isEqualTo(new int[] { 1, 2, 3, 4, 5 });
   }

   @Test
   public void testLoadSaveModel() throws Exception {
      Model model = new Model();
      model.solverType = SolverType.L2_LR;
      model.bias = 243.0;
      model.w = new double[] { 1, 2, 0, 4, 5, 6, 0, 7, 8, 9, 1, 723.99 };
      model.label = new int[] { 1, Integer.MAX_VALUE, 2 };
      model.nr_feature = model.w.length / model.label.length - 1;
      model.nr_class = model.label.length;

      File tempFile = File.createTempFile("linear", "modeltest");
      tempFile.deleteOnExit();
      Linear.saveModel(tempFile.getAbsolutePath(), model);
      Model loadedModel = Linear.loadModel(tempFile.getAbsolutePath());

      assertThat(loadedModel.w).isEqualTo(model.w);
      assertThat(loadedModel).isEqualTo(model);
   }

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

      Parameter param = new Parameter(SolverType.L2_LR, 100, 0.1);
      Model model = Linear.train(prob, param);

      int i = 0;
      for ( int value : prob.y ) {
         int prediction = Linear.predict(model, prob.x[i]);
         assertThat(prediction).isEqualTo(value);
         i++;
      }
   }
}
