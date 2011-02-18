package de.bwaldvogel.liblinear;

import static org.fest.assertions.Assertions.assertThat;
import static org.fest.assertions.Fail.fail;

import org.junit.Test;


public class ArrayPointerTest {

    @Test
    public void testGetIntArrayPointer() {
        int[] foo = new int[] {1, 2, 3, 4, 6};
        IntArrayPointer pFoo = new IntArrayPointer(foo, 2);
        assertThat(pFoo.get(0)).isEqualTo(3);
        assertThat(pFoo.get(1)).isEqualTo(4);
        assertThat(pFoo.get(2)).isEqualTo(6);
        try {
            pFoo.get(3);
            fail("ArrayIndexOutOfBoundsException expected");
        } catch (ArrayIndexOutOfBoundsException e) {}
    }

    @Test
    public void testSetIntArrayPointer() {
        int[] foo = new int[] {1, 2, 3, 4, 6};
        IntArrayPointer pFoo = new IntArrayPointer(foo, 2);
        pFoo.set(2, 5);
        assertThat(foo).isEqualTo(new int[] {1, 2, 3, 4, 5});
        try {
            pFoo.set(3, 0);
            fail("ArrayIndexOutOfBoundsException expected");
        } catch (ArrayIndexOutOfBoundsException e) {}
    }

    @Test
    public void testGetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        DoubleArrayPointer pFoo = new DoubleArrayPointer(foo, 2);
        assertThat(pFoo.get(0)).isEqualTo(3);
        assertThat(pFoo.get(1)).isEqualTo(4);
        assertThat(pFoo.get(2)).isEqualTo(6);
        try {
            pFoo.get(3);
            fail("ArrayIndexOutOfBoundsException expected");
        } catch (ArrayIndexOutOfBoundsException e) {}
    }

    @Test
    public void testSetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        DoubleArrayPointer pFoo = new DoubleArrayPointer(foo, 2);
        pFoo.set(2, 5);
        assertThat(foo).isEqualTo(new double[] {1, 2, 3, 4, 5});
        try {
            pFoo.set(3, 0);
            fail("ArrayIndexOutOfBoundsException expected");
        } catch (ArrayIndexOutOfBoundsException e) {}
    }
}
