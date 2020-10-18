package de.bwaldvogel.liblinear;

import static org.assertj.core.api.Assertions.*;

import org.junit.jupiter.api.Test;


class ArrayPointerTest {

    @Test
    void testGetIntArrayPointer() {
        int[] foo = new int[] {1, 2, 3, 4, 6};
        IntArrayPointer pFoo = new IntArrayPointer(foo, 2);
        assertThat(pFoo.get(0)).isEqualTo(3);
        assertThat(pFoo.get(1)).isEqualTo(4);
        assertThat(pFoo.get(2)).isEqualTo(6);

        assertThatExceptionOfType(ArrayIndexOutOfBoundsException.class)
            .isThrownBy(() -> pFoo.get(3));
    }

    @Test
    void testSetIntArrayPointer() {
        int[] foo = new int[] {1, 2, 3, 4, 6};
        IntArrayPointer pFoo = new IntArrayPointer(foo, 2);
        pFoo.set(2, 5);
        assertThat(foo).isEqualTo(new int[] {1, 2, 3, 4, 5});

        assertThatExceptionOfType(ArrayIndexOutOfBoundsException.class)
            .isThrownBy(() -> pFoo.set(3, 0));
    }

    @Test
    void testGetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        DoubleArrayPointer pFoo = new DoubleArrayPointer(foo, 2);
        assertThat(pFoo.get(0)).isEqualTo(3);
        assertThat(pFoo.get(1)).isEqualTo(4);
        assertThat(pFoo.get(2)).isEqualTo(6);

        assertThatExceptionOfType(ArrayIndexOutOfBoundsException.class)
            .isThrownBy(() -> pFoo.get(3));
    }

    @Test
    void testSetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        DoubleArrayPointer pFoo = new DoubleArrayPointer(foo, 2);
        pFoo.set(2, 5);
        assertThat(foo).isEqualTo(new double[] {1, 2, 3, 4, 5});

        assertThatExceptionOfType(ArrayIndexOutOfBoundsException.class)
            .isThrownBy(() -> pFoo.set(3, 0));
    }
}
