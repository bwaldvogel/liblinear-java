package de.bwaldvogel.liblinear;

class Heap {

    enum HeapType {
        MIN, MAX,
    }


    private       int       size;
    private final HeapType  type;
    private final Feature[] a;

    Heap(int max_size, HeapType type) {
        this.size = 0;
        this.a = new Feature[max_size];
        this.type = type;
    }

    private boolean cmp(Feature left, Feature right) {
        if (this.type == HeapType.MIN) {
            return left.getValue() > right.getValue();
        } else {
            return left.getValue() < right.getValue();
        }
    }

    int size() {
        return size;
    }

    void push(Feature node) {
        a[size] = node;
        size++;
        int i = size - 1;
        while (i > 0) {
            int p = (i - 1) / 2;
            if (cmp(a[p], a[i])) {
                Linear.swap(a, i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    void pop() {
        size--;
        a[0] = a[size];
        int i = 0;
        while (i * 2 + 1 < size) {
            int l = i * 2 + 1;
            int r = i * 2 + 2;
            if (r < size && cmp(a[l], a[r])) {
                l = r;
            }
            if (cmp(a[i], a[l])) {
                Linear.swap(a, i, l);
                i = l;
            } else {
                break;
            }
        }
    }

    Feature top() {
        return this.a[0];
    }

}
