package liblinear;

import org.junit.Test;
import static org.fest.assertions.Assertions.assertThat;


public class FeatureNodeTest {

   @Test(expected = IllegalArgumentException.class)
   public void testConstructorIndexZero() {
      new FeatureNode(0, 0);
   }

   @Test(expected = IllegalArgumentException.class)
   public void testConstructorIndexNegative() {
      new FeatureNode(-1, 0);
   }

   public void testConstructorHappy() {
      FeatureNode fn = new FeatureNode(25, 27.39);
      assertThat(fn.index).isEqualTo(25);
      assertThat(fn.value).isEqualTo(27.39);

      fn = new FeatureNode(1, -0.22222);
      assertThat(fn.index).isEqualTo(1);
      assertThat(fn.value).isEqualTo(-0.22222);
   }
}
