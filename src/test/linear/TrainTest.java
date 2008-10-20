package linear;

import static org.fest.assertions.Assertions.assertThat;
import static linear.Linear.NL;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;


public class TrainTest {

   @Test
   public void testReadProblem() throws Exception {

      File file = File.createTempFile("svm", "test");
      file.deleteOnExit();

      Collection<String> lines = new ArrayList<String>();
      lines.add("1 1:1  3:1  4:1   6:1");
      lines.add("2 2:1  3:1  5:1   7:1");
      lines.add("1 3:1  5:1");
      lines.add("1 1:1  4:1  7:1");
      lines.add("2 4:1  5:1  7:1");
      BufferedWriter writer = new BufferedWriter(new FileWriter(file));
      try {
         for ( String line : lines )
            writer.append(line).append(NL);
      }
      finally {
         writer.close();
      }

      Train train = new Train();
      train.readProblem(file.getAbsolutePath());

      Problem prob = train.getProblem();
      assertThat(prob.y).hasSize(lines.size());
      assertThat(prob.y).isEqualTo(new int[] { 1, 2, 1, 1, 2 });
      assertThat(prob.n).isEqualTo(8);
      System.out.println(prob.l);

   }
}
