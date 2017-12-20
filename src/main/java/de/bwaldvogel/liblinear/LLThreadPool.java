package de.bwaldvogel.liblinear;

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
class LLThreadPool {
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
    this.pool = new ThreadPoolExecutor(0, threadCount, 15, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>(),
        new ThreadFactory() {
          private final ThreadFactory defaultFactory = Executors.defaultThreadFactory();

          public Thread newThread(Runnable r) {
            Thread thread = defaultFactory.newThread(r);
            thread.setDaemon(true);
            return thread;
          }
        });
  }

  /**
   * Define a functional class since we're targeting Java 1.7
   */
  public static abstract class RangeConsumer {
    public abstract void run(int start, int endExclusive);
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
