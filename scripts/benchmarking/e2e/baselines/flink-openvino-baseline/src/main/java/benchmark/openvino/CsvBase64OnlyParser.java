package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.util.Collector;

/**
 * Parses the single base64 column out of each CSV line and stamps the record's ingestion time.
 *
 * To keep the framework comparison fair, ingestion is paced at a fixed rate matching the NES / TorchServe
 * TCP sources (MOCK_TUPLE_RATE): without this, Flink's FileSource is a backpressure-driven firehose while
 * the NES/TorchServe sources emit at a controlled tuples/second. The pacing is cumulative (release the
 * i-th record no earlier than start + i*interval), mirroring the NES TCPDataServer SendRateLimiter, and
 * {@code e2eStartNs} is stamped AFTER pacing so it is the true (paced) arrival time -- the same point NES
 * stamps its ingestion timestamp. With parallelism > 1 the rate is split across subtasks so the aggregate
 * ingestion rate matches the single NES source.
 */
public final class CsvBase64OnlyParser extends RichFlatMapFunction<String, FrameRecord> {
    private final int base64Column;
    private final boolean hasHeader;
    private final double ratePerSecond;   // <= 0 => unbounded (no pacing)
    private final int inferenceLanes;     // engines that must be ready before pacing starts

    private transient boolean firstLine;
    private transient long intervalNs;    // per-subtask inter-record interval; 0 => unbounded
    private transient long startNs;
    private transient long emitted;
    /// Column-slicing cost only. The timer starts AFTER the pacing sleep and the arrival stamp,
    /// so it measures this stage's work rather than the rate limiter holding the record back.
    private transient LatencyStats parseStats;

    public CsvBase64OnlyParser(int base64Column, boolean hasHeader, double ratePerSecond, int inferenceLanes) {
        this.base64Column = base64Column;
        this.hasHeader = hasHeader;
        this.ratePerSecond = ratePerSecond;
        this.inferenceLanes = inferenceLanes;
        EngineReadiness.expect(inferenceLanes);
    }

    @Override
    public void open(OpenContext openContext) {
        firstLine = true;
        parseStats = new LatencyStats();
        emitted = 0L;
        startNs = 0L;
        int subtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        double perSubtask = ratePerSecond > 0 ? ratePerSecond / Math.max(1, subtasks) : 0.0;
        intervalNs = perSubtask > 0 ? (long) (1_000_000_000.0 / perSubtask) : 0L;
    }

    @Override
    public void flatMap(String line, Collector<FrameRecord> out) throws Exception {
        if (line == null || line.isBlank()) {
            return;
        }

        if (hasHeader && firstLine) {
            firstLine = false;
            return;
        }
        firstLine = false;

        // Fixed-rate ingestion pacing (see class doc). Sleep until this record's scheduled slot,
        // then stamp arrival, so latency is measured against a controlled arrival rate.
        if (intervalNs > 0L) {
            if (emitted == 0L) {
                // Do not start the clock until every lane can serve: see EngineReadiness.
                EngineReadiness.awaitAll();
                startNs = System.nanoTime();
            }
            long sleepNs = (startNs + emitted * intervalNs) - System.nanoTime();
            if (sleepNs > 0L) {
                Thread.sleep(sleepNs / 1_000_000L, (int) (sleepNs % 1_000_000L));
            }
        }
        emitted++;

        long e2eStartNs = System.nanoTime();

        FrameRecord parsed = new FrameRecord(sliceColumn(line), e2eStartNs);
        parseStats.record(e2eStartNs, System.nanoTime());
        out.collect(parsed);
    }

    @Override
    public void close() {
        if (parseStats != null) {
            StatsCsvWriter.record("parse", parseStats);
        }
    }

    /**
     * Slices one column out directly instead of running a CSV parser over the line.
     *
     * These frame CSVs are "timestamp,base64jpeg" -- base64's alphabet contains no comma and the
     * rows carry no quoting -- so a general parser buys nothing and costs a great deal: building a
     * CSVFormat, a Lexer and a StringReader per record, then tokenising a 360 KB line character by
     * character and materialising a second 360 KB String. Measured on the Pi that was ~67 ms per
     * frame, which capped the whole job at 14.5 frames/s while its two inference lanes could serve
     * ~58/s -- the baseline's reported throughput was measuring its CSV parser, not its serving
     * path. {@link SlidingWindowSampleParser#parseColumn} already avoided this for the same reason.
     */
    private String sliceColumn(String line) {
        int start = 0;
        for (int column = 0; column < base64Column; column++) {
            start = line.indexOf(',', start);
            if (start < 0) {
                throw new IllegalArgumentException("Line has no column " + base64Column);
            }
            start++;
        }
        int end = line.indexOf(',', start);
        if (end < 0) {
            end = line.length();
        }
        return line.substring(start, end).trim();
    }
}
