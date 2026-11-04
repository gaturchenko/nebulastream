package benchmark.openvino;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Opt-in per-record timeline, for finding where a slow window waited and for cutting a run's start-up.
 *
 * <p>The comparison row only carries per-stage distributions, and on CWRU the end-to-end tail is
 * far longer than the inference tail, so the difference is time spent waiting somewhere -- which a
 * distribution cannot place. This records, per window, when its newest sample was due, when it
 * actually arrived (the latency clock start), when the window left the windowing stage, which lane
 * served it and when that lane started and finished it, and when the result reached the sink:
 *
 * <pre>
 *   arrival   - scheduled  : the paced source running late (its thread starved of CPU)
 *   infer_start - emitted  : waiting in the lane's input queue, behind earlier windows
 *   infer_end - infer_start: the inference call itself
 * </pre>
 *
 * <p>Only the windowed job numbers its records, so only its rows carry every stage. The object-detection
 * job's records have no sequence number ({@code seq} -1) and get just their arrival and sink times, under
 * a negative key of their own -- enough to recompute end-to-end percentiles over any part of the run.
 *
 * <p>Enabled by setting {@code FLINK_WINDOW_TRACE} to the output CSV path; unset, every hook is a
 * single branch. All timestamps are {@link System#nanoTime()}, comparable because the job runs in
 * one MiniCluster JVM (as {@link StatsCsvWriter} also assumes), and the same clock as the JVM's
 * {@code -Xlog} {@code timenanos} decorator, so GC pauses can be laid over the trace.
 */
public final class WindowTrace {
    private static final String PATH = System.getenv("FLINK_WINDOW_TRACE");

    private static final int SCHEDULED = 0;
    private static final int ARRIVAL = 1;
    private static final int EMITTED = 2;
    private static final int LANE = 3;
    private static final int INFER_START = 4;
    private static final int INFER_END = 5;
    private static final int SINK = 6;

    /** window sequence number -> the slots above; each stage fills its own, from its own thread. */
    private static final Map<Long, long[]> ROWS = new ConcurrentHashMap<>();
    /** Keys for records without a sequence number: -1, -2, ... in sink order. */
    private static final AtomicLong UNSEQUENCED = new AtomicLong();

    private WindowTrace() {}

    public static boolean enabled() {
        return PATH != null && !PATH.isBlank();
    }

    public static void emitted(long seq, long scheduledNs, long arrivalNs, long emittedNs) {
        long[] row = row(seq);
        row[SCHEDULED] = scheduledNs;
        row[ARRIVAL] = arrivalNs;
        row[EMITTED] = emittedNs;
    }

    public static void inferred(long seq, int lane, long startNs, long endNs) {
        if (seq < 0) {
            return;  // an unnumbered record: only delivered() can key it
        }
        long[] row = row(seq);
        row[LANE] = lane;
        row[INFER_START] = startNs;
        row[INFER_END] = endNs;
    }

    public static void delivered(long seq, long arrivalNs, long sinkNs) {
        long[] row = row(seq >= 0 ? seq : UNSEQUENCED.decrementAndGet());
        row[ARRIVAL] = arrivalNs;
        row[SINK] = sinkNs;
    }

    /** Writes one row per record, numbered windows in window order. Call after env.execute() returns. */
    public static void write() throws IOException {
        if (!enabled()) {
            return;
        }
        Path path = Path.of(PATH);
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("seq,scheduled_ns,arrival_ns,emitted_ns,lane,infer_start_ns,infer_end_ns,sink_ns");
            out.newLine();
            for (Map.Entry<Long, long[]> entry : new TreeMap<>(ROWS).entrySet()) {
                long[] row = entry.getValue();
                out.write(entry.getKey().toString());
                for (long value : row) {
                    out.write(',');
                    out.write(Long.toString(value));
                }
                out.newLine();
            }
        }
        System.out.println("Wrote record trace (" + ROWS.size() + " records): " + path);
    }

    private static long[] row(long seq) {
        return ROWS.computeIfAbsent(seq, ignored -> new long[SINK + 1]);
    }
}
