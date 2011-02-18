package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.Linear.swap;
import static org.fest.assertions.Assertions.assertThat;

import java.util.Random;

import org.junit.Test;


public class ArraySorterTest {

    private Random random = new Random();

    private void assertDescendingOrder(double[] array) {
        double before = array[0];
        for (double d : array) {
            // accept that case
            if (d == 0.0 && before == -0.0) continue;

            assertThat(d).isLessThanOrEqualTo(before);
            before = d;
        }
    }

    private void shuffleArray(double[] array) {

        for (int i = 0; i < array.length; i++) {
            int j = random.nextInt(array.length);
            swap(array, i, j);
        }
    }

    @Test
    public void testReversedMergesort() {

        for (int k = 1; k <= 16 * 8096; k *= 2) {
            // create random array
            double[] array = new double[k];
            for (int i = 0; i < array.length; i++) {
                array[i] = random.nextDouble();
            }

            ArraySorter.reversedMergesort(array);
            assertDescendingOrder(array);
        }
    }

    @Test
    public void testReversedMergesortWithMeanValues() {
        double[] array = new double[] {1.0, -0.0, -1.1, 2.0, 3.0, 0.0, 4.0, -0.0, 0.0};
        shuffleArray(array);
        ArraySorter.reversedMergesort(array);
        assertDescendingOrder(array);
    }
}
