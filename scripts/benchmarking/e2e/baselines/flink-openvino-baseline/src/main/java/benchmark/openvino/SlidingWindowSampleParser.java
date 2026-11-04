package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.util.Collector;

import java.util.concurrent.locks.LockSupport;

/**
 * Turns a CSV of scalar samples into the sliding windows the model consumes, pacing ingestion at a
 * fixed rate. This is the sensor-stream counterpart of {@link CsvBase64OnlyParser}: where that one
 * emits one record per frame, one row here is a single sample and a model input is built from
 * {@code windowSamples} consecutive samples, re-emitted every {@code advanceSamples}.
 *
 * It mirrors the aggregation of the NES query
 *
 * <pre>
 *   ARRAY_AGG(value) ... WINDOW SLIDING(timestamp, SIZE s, ADVANCE BY a)
 * </pre>
 *
 * on a source paced at a fixed tuples/second, where SIZE and ADVANCE are exactly
 * {@code windowSamples} and {@code advanceSamples} samples wide.
 *
 * <p>The window is kept in a ring buffer and copied out once per advance, which costs O(1) per
 * sample instead of the O(size/advance) an event-time sliding window operator pays to keep a sample
 * in every pane it belongs to (20 panes for CWRU, 3 for MUSAN). The comparison deliberately gives
 * the baseline this cheaper hand-rolled window: the measured gap is then a lower bound on the
 * framework's overhead rather than an artifact of Flink's windowing machinery.
 *
 * <p>Like the NES query's {@code MAX(ingestion_time)}, a window's latency clock starts at the
 * arrival of its NEWEST sample -- the window is emitted the moment that sample is ingested, so the
 * stamp is taken then. Only complete windows are emitted; the first {@code windowSamples - 1}
 * samples only fill the buffer.
 */
public final class SlidingWindowSampleParser
        extends RichFlatMapFunction<String, PreprocessedFrameRecord> {

    private final int valueColumn;
    private final boolean hasHeader;
    private final double ratePerSecond;   // <= 0 => unbounded (no pacing)
    private final int windowSamples;
    private final int advanceSamples;
    private final float scaleDivisor;     // <= 0 => no scaling
    private final int inferenceLanes;     // engines that must be ready before pacing starts

    private transient boolean firstLine;
    private transient long intervalNs;    // per-subtask inter-sample interval; 0 => unbounded
    private transient long startNs;
    private transient long ingested;

    private transient float[] ring;
    private transient int writeIndex;     // next slot to write
    private transient int filled;         // samples buffered so far, capped at windowSamples
    private transient int sinceLastEmit;
    private transient long windowSeq;     // emitted windows so far, the WindowTrace key
    /// Window materialisation only -- the ring-buffer copy that builds one model input. Timed per
    /// EMITTED WINDOW, not per sample: at the 48 kHz CWRU rate a nanoTime() per sample would cost
    /// more than the thing being measured and would perturb the pacing it sits next to.
    private transient LatencyStats stats;

    public SlidingWindowSampleParser(
            int valueColumn,
            boolean hasHeader,
            double ratePerSecond,
            int windowSamples,
            int advanceSamples,
            float scaleDivisor,
            int inferenceLanes) {
        if (windowSamples <= 0) {
            throw new IllegalArgumentException("windowSamples must be > 0, got " + windowSamples);
        }
        if (advanceSamples <= 0 || advanceSamples > windowSamples) {
            throw new IllegalArgumentException(
                    "advanceSamples must be in (0, windowSamples], got " + advanceSamples);
        }
        this.valueColumn = valueColumn;
        this.hasHeader = hasHeader;
        this.ratePerSecond = ratePerSecond;
        this.windowSamples = windowSamples;
        this.advanceSamples = advanceSamples;
        this.scaleDivisor = scaleDivisor;
        this.inferenceLanes = inferenceLanes;
        EngineReadiness.expect(inferenceLanes);
    }

    @Override
    public void open(OpenContext openContext) {
        firstLine = true;
        ingested = 0L;
        startNs = 0L;
        int subtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        double perSubtask = ratePerSecond > 0 ? ratePerSecond / Math.max(1, subtasks) : 0.0;
        intervalNs = perSubtask > 0 ? (long) (1_000_000_000.0 / perSubtask) : 0L;

        stats = new LatencyStats();
        ring = new float[windowSamples];
        writeIndex = 0;
        filled = 0;
        sinceLastEmit = 0;
        windowSeq = 0L;
    }

    @Override
    public void flatMap(String line, Collector<PreprocessedFrameRecord> out) {
        if (line == null || line.isBlank()) {
            return;
        }
        if (hasHeader && firstLine) {
            firstLine = false;
            return;
        }
        firstLine = false;

        float value = parseColumn(line);

        // Cumulative fixed-rate pacing, identical in shape to the NES TCP source's rate limiter:
        // the i-th sample is released no earlier than start + i*interval, and a slot already in the
        // past is never slept on, so a source that falls behind catches up instead of drifting.
        // parkNanos (not Thread.sleep) because these intervals are tens of microseconds -- 20.8 us at
        // the 48 kHz CWRU rate -- which millisecond-granularity sleeping cannot express at all.
        long scheduledNs = 0L;
        if (intervalNs > 0L) {
            if (ingested == 0L) {
                // Do not start the clock until every lane can serve: see EngineReadiness.
                EngineReadiness.awaitAll();
                startNs = System.nanoTime();
            }
            long targetNs = startNs + ingested * intervalNs;
            scheduledNs = targetNs;
            long remainingNs = targetNs - System.nanoTime();
            while (remainingNs > 0L) {
                LockSupport.parkNanos(remainingNs);
                remainingNs = targetNs - System.nanoTime();
            }
        }
        ingested++;

        long arrivalNs = System.nanoTime();

        ring[writeIndex] = value;
        writeIndex = (writeIndex + 1) % windowSamples;
        if (filled < windowSamples) {
            filled++;
        }
        sinceLastEmit++;

        if (filled == windowSamples && sinceLastEmit >= advanceSamples) {
            sinceLastEmit = 0;
            long startNs = System.nanoTime();
            long seq = windowSeq++;
            PreprocessedFrameRecord window = new PreprocessedFrameRecord(copyWindow(), arrivalNs, seq);
            long emittedNs = System.nanoTime();
            stats.record(startNs, emittedNs);
            if (WindowTrace.enabled()) {
                WindowTrace.emitted(seq, scheduledNs, arrivalNs, emittedNs);
            }
            out.collect(window);
        }
    }

    @Override
    public void close() {
        if (stats != null) {
            StatsCsvWriter.record("preprocess", stats);
        }
    }

    /** Oldest-to-newest copy of the ring buffer: writeIndex is both the next slot and the oldest sample. */
    private float[] copyWindow() {
        float[] window = new float[windowSamples];
        int tail = windowSamples - writeIndex;
        System.arraycopy(ring, writeIndex, window, 0, tail);
        System.arraycopy(ring, 0, window, tail, writeIndex);
        return window;
    }

    /**
     * Reads one numeric column. These CSVs hold plain numbers with no quoting or embedded commas,
     * so the column is sliced out directly -- at 48 000 rows/s a full CSV parser per row would eat
     * into the pacing budget.
     */
    private float parseColumn(String line) {
        int start = 0;
        for (int column = 0; column < valueColumn; column++) {
            start = line.indexOf(',', start);
            if (start < 0) {
                throw new IllegalArgumentException(
                        "Line has no column " + valueColumn + ": " + line);
            }
            start++;
        }
        int end = line.indexOf(',', start);
        if (end < 0) {
            end = line.length();
        }

        float value = Float.parseFloat(line.substring(start, end).trim());
        return scaleDivisor > 0 ? value / scaleDivisor : value;
    }
}
