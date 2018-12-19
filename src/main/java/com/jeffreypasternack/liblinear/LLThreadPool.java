package com.jeffreypasternack.liblinear;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;


/**
 * Wraps a ThreadPoolExecutor and provides utility methods for multiplexing operations and accumulating results
 */
class LLThreadPool implements AutoCloseable {
  /**
   * Sets the maximum number of items/iterations that should be handled by a single job.  A higher limit reduces the
   * number of "batches"/jobs that need to be enqueued and joined, but also increases time that some threads will idle
   * while others are finishing the last few jobs.
   */
  private static final int PER_JOB_MAX_CHUNK_SIZE = 10000;

  private final int threadCount;
  private ThreadPoolExecutor pool;

  public LLThreadPool(int threadCount) {
    this.threadCount = threadCount;
    this.pool =
        new ThreadPoolExecutor(threadCount, threadCount, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>(),
            new ThreadFactory() {
              private final ThreadFactory defaultFactory = Executors.defaultThreadFactory();

              public Thread newThread(Runnable r) {
                Thread thread = defaultFactory.newThread(r);
                thread.setDaemon(true);
                return thread;
              }
            });
  }

  @Override
  public void close() {
    this.pool.shutdown();
    this.pool = null;
  }

  private static class ThreadLocalDoubleArray extends ThreadLocal<double[]> {
    private final int length;

    public ThreadLocalDoubleArray(int length) {
      this.length = length;
    }

    @Override
    public double[] get() {
      double[] res = super.get();
      Arrays.fill(res, 0);
      return res;
    }

    @Override
    protected double[] initialValue() {
      return new double[this.length];
    }
  }

  /**
   * Define a functional class since we're targeting Java 1.7
   */
  public static abstract class RangeConsumer {
    public abstract void run(int start, int endExclusive);
  }

  /**
   * Specialized RangeConsumer variant for the common case where a RangeConsumer uses a thread-local accumulator array
   * that is ultimately summed into a global result array.
   */
  public static abstract class RangeConsumerWithAccumulatorArray extends RangeConsumer {
    private final ThreadLocalDoubleArray threadLocalAccumulator;
    private final double[] globalAccumulator;
    private final Object synchronizer = new Object();

    public RangeConsumerWithAccumulatorArray(double[] globalAccumulator) {
      this.threadLocalAccumulator = new ThreadLocalDoubleArray(globalAccumulator.length);
      this.globalAccumulator = globalAccumulator;
    }

    public final void run(int start, int endExclusive) {
      double[] accumulator = threadLocalAccumulator.get();
      run(start, endExclusive, accumulator);
      synchronized (synchronizer) {
        for (int i = 0; i < globalAccumulator.length; i++) {
          globalAccumulator[i] += accumulator[i];
        }
      }
    }

    public abstract void run(int start, int endExclusive, double[] accumulator);
  }

  private static class RangeConsumerRunnable implements Runnable {
    private final RangeConsumer consumer;
    private final int start;
    private final int endExclusive;

    public RangeConsumerRunnable(RangeConsumer consumer, int start, int endExclusive) {
      this.consumer = consumer;
      this.start = start;
      this.endExclusive = endExclusive;
    }

    @Override
    public void run() {
      consumer.run(this.start, this.endExclusive);
    }
  }

  public void execute(RangeConsumer consumer, int count) {
    if (count == 0) {
      return;
    }

    int chunkSize = Math.min(PER_JOB_MAX_CHUNK_SIZE, count / threadCount);
    int jobs = chunkSize >= 1 ? count / chunkSize : 0; // for the moment, pretend 0/0 == 0
    int remainderStart = jobs * chunkSize;

    Future<?>[] futures = new Future<?>[jobs + (count > remainderStart ? 1 : 0)];

    for (int i = 0; i < jobs; i++) {
      int start = i * chunkSize;
      futures[i] = pool.submit(new RangeConsumerRunnable(consumer, start, start + chunkSize));
    }

    // schedule the last "chunk"
    if (count > remainderStart) {
      futures[futures.length - 1] = pool.submit(new RangeConsumerRunnable(consumer, remainderStart, count));
    }

    try {
      for (Future<?> future : futures) {
        future.get(); // join on each future
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt(); // re-interrupt our thread
    } catch (ExecutionException e) {
      throw new RuntimeException(e); // re-throw as a runtime exception to evade declaring a checked exception
    }
  }
}
