package benchmark.openvino;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Holds the paced source back until every inference lane has a compiled, warmed-up OpenVINO engine.
 *
 * <p>Without this the comparison is unfair to Flink. Its operators open concurrently, so the source
 * starts releasing samples at the configured rate while the inference operators are still inside
 * {@code compile_model} -- tens of seconds for a large IR. The windows produced in that span queue
 * up, and the lanes then work through the backlog back-to-back, which inflates both the end-to-end
 * latency and, because the lanes now overlap instead of interleaving, the per-inference time. The
 * shorter the run, the more the transient dominates: the same MUSAN job measured 168 ms per
 * inference over 583 s and 310 ms over 60 s.
 *
 * <p>NES has no equivalent transient -- its model is bound while the query is deployed, before the
 * source streams -- so gating the source here measures the same steady state on both sides rather
 * than charging Flink for a warm-up its counterpart never pays.
 *
 * <p>All operators of a run share one JVM (the job runs on a local MiniCluster), so a static latch
 * is enough. The await is bounded: if an engine never reports ready the source starts anyway, and
 * the run produces a visibly skewed number instead of hanging forever.
 */
final class EngineReadiness {
    private static final long DEFAULT_TIMEOUT_MS = 600_000L;

    private static CountDownLatch latch;

    private EngineReadiness() {}

    /** Declares how many engines a run will open. The first caller wins; later calls with the same count are no-ops. */
    static synchronized void expect(int engines) {
        if (latch == null) {
            latch = new CountDownLatch(Math.max(0, engines));
        }
    }

    private static synchronized CountDownLatch latch() {
        return latch;
    }

    /** Called by each inference lane once its engine is compiled AND warmed up. */
    static void ready() {
        CountDownLatch current = latch();
        if (current != null) {
            current.countDown();
        }
    }

    /** Called by the source before the first sample is released. */
    static void awaitAll() {
        CountDownLatch current = latch();
        if (current == null) {
            return;
        }
        try {
            if (!current.await(DEFAULT_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                System.err.println(
                        "WARNING: inference engines were not ready after " + DEFAULT_TIMEOUT_MS
                                + " ms; starting the source anyway. The reported latency will include "
                                + "engine startup.");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
