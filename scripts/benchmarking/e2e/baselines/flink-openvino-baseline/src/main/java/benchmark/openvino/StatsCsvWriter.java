package benchmark.openvino;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.Map;

/**
 * Accumulates per-scope latency stats during a run and, once the job finishes, emits ONE wide row
 * per run whose columns line up with the NES / TorchServe comparison tables produced by
 * process_results.py / process_torchserve_results.py. That way a Flink baseline row drops straight
 * into the OSU-RGB framework comparison, substituting the NES-UDF case.
 *
 * Two scopes are recorded: "e2e" (parse+preprocess+infer, from E2eStatsMapFunction) and
 * "inference" (pure OpenVINO predict, from OpenVinoFrameMapFunction). Subtask stats are merged, so
 * the row is the aggregate across Flink parallelism. Call configureRun() before the job and
 * writeWideRow() after env.execute() returns.
 */
public final class StatsCsvWriter {
    private static final Object LOCK = new Object();

    /** scope ("e2e" | "inference") -> stats merged across parallel subtasks for the current run. */
    private static final Map<String, LatencyStats> SCOPES = new HashMap<>();

    // Run identity, matching the NES/TorchServe comparison columns. Set by the job; safe defaults.
    private static volatile String queryName = "flink";
    private static volatile String paramName = "baseline";
    private static volatile String paramValue = "flink";
    private static volatile String repetition = "0";

    private static final String HEADER = String.join(",",
            "query_name",
            "inference_config_param_name",
            "inference_config_param_value",
            "repetition",
            "flink_records",
            "end_to_end_duration_us",    // wall-clock span first-arrival -> last-output, cf. NES
            "end_to_end_throughput",     // records/s, cf. NES end_to_end_throughput
            "end_to_end_latency_us",     // mean e2e latency, cf. NES end_to_end_latency_us
            "e2e_latency_ms_mean",       // cf. TorchServe e2e_latency_ms_*
            "e2e_latency_ms_p50",
            "e2e_latency_ms_p95",
            "e2e_latency_ms_p99",
            "inference_latency_ms_mean", // pure-inference distribution (Flink-specific detail)
            "inference_latency_ms_p50",
            "inference_latency_ms_p95",
            "inference_latency_ms_p99",
            // Java<->native split of that same inference call, as MEANS (only means compose:
            // native_predict covers the two OpenVINO tensor memcpys plus the model, native_infer
            // only the model, and inference_latency_ms_mean minus native_predict is the JNI array
            // access). Zero when the native library predates the timing counters.
            "native_predict_ms_mean",
            "native_infer_ms_mean",
            // Per-stage CPU work, as MEANS. Only means compose: e2e spans all three stages PLUS
            // the waiting between them, so e2e_mean minus these three is queueing/backpressure,
            // which is what separates "Flink does expensive per-frame work" from "Flink is
            // backlogged". Percentiles would not subtract meaningfully.
            "parse_latency_ms_mean",
            "preprocess_latency_ms_mean"
    );

    /// Native counters at the moment steady state began, subtracted from the totals below.
    ///
    /// Each lane runs a throwaway warm-up predict in open(), which the Java-side LatencyStats does
    /// not record but the process-wide native counters do -- and a first inference is slow enough
    /// to push the native mean ABOVE the Java mean, which is impossible for nested spans and makes
    /// the subtraction meaningless. The source does not emit until every lane has reported ready
    /// (see EngineReadiness), so the first mapped record is a safe point at which all warm-ups are
    /// already counted.
    private static final AtomicBoolean baselineTaken = new AtomicBoolean(false);
    private static volatile long[] nativeBaseline = {0L, 0L, 0L};

    private StatsCsvWriter() {}

    /** Called on the first steady-state record; snapshots the native counters once. */
    public static void markSteadyState() {
        if (baselineTaken.compareAndSet(false, true)) {
            try {
                long[] t = NativeOpenVinoBridge.nativeTimings();
                if (t != null && t.length >= 3) {
                    nativeBaseline = t;
                }
            } catch (UnsatisfiedLinkError | RuntimeException e) {
                // An older native library has no counters; the breakdown columns stay zero.
            }
        }
    }

    /** Set the comparison-key columns (query/config/repetition) before the job runs. */
    public static void configureRun(String query, String pName, String pValue, String rep) {
        if (query != null && !query.isBlank()) {
            queryName = query;
        }
        if (pName != null && !pName.isBlank()) {
            paramName = pName;
        }
        if (pValue != null && !pValue.isBlank()) {
            paramValue = pValue;
        }
        if (rep != null && !rep.isBlank()) {
            repetition = rep;
        }
    }

    /** Buffer one subtask's stats for a scope, merging across parallel subtasks. */
    public static void record(String scope, LatencyStats stats) {
        if (scope == null || stats == null) {
            return;
        }
        synchronized (LOCK) {
            SCOPES.computeIfAbsent(scope, ignored -> new LatencyStats()).merge(stats);
        }
    }

    /** Write the single wide row for this run. Call once, after env.execute() returns. */
    public static void writeWideRow(String statsCsvPath) throws IOException {
        if (statsCsvPath == null || statsCsvPath.isBlank()) {
            return;
        }

        Path path = Path.of(statsCsvPath);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        synchronized (LOCK) {
            LatencyStats e2e = SCOPES.get("e2e");
            LatencyStats inference = SCOPES.get("inference");
            LatencyStats parse = SCOPES.get("parse");
            LatencyStats preprocess = SCOPES.get("preprocess");

            boolean writeHeader = !Files.exists(path) || Files.size(path) == 0L;
            try (BufferedWriter writer = Files.newBufferedWriter(
                    path,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            )) {
                if (writeHeader) {
                    writer.write(HEADER);
                    writer.newLine();
                }
                writer.write(formatRow(e2e, inference, parse, preprocess));
                writer.newLine();
            }
            SCOPES.clear();
        }
    }

    private static String formatRow(LatencyStats e2e, LatencyStats inference,
                                    LatencyStats parse, LatencyStats preprocess) {
        long records = e2e != null ? e2e.count() : (inference != null ? inference.count() : 0L);
        double e2eDurationUs = e2e != null ? e2e.elapsedNs() / 1_000.0 : 0.0;
        double e2eThroughput = e2e != null ? e2e.throughputRecordsPerSecond() : 0.0;
        double e2eLatencyUs = e2e != null ? e2e.averageLatencyNs() / 1_000.0 : 0.0;

        return String.join(",",
                csv(queryName),
                csv(paramName),
                csv(paramValue),
                csv(repetition),
                Long.toString(records),
                formatDouble(e2eDurationUs),
                formatDouble(e2eThroughput),
                formatDouble(e2eLatencyUs),
                msMean(e2e),
                msPercentile(e2e, 50.0),
                msPercentile(e2e, 95.0),
                msPercentile(e2e, 99.0),
                msMean(inference),
                msPercentile(inference, 50.0),
                msPercentile(inference, 95.0),
                msPercentile(inference, 99.0),
                formatDouble(nativeMeanMs(1)),
                formatDouble(nativeMeanMs(2)),
                msMean(parse),
                msMean(preprocess)
        );
    }

    /**
     * Mean of one native timing total, in milliseconds: index 1 = whole native predict, 2 = the
     * model call alone. Returns 0 if the loaded native library has no timing counters, which keeps
     * an older libflink_openvino_bridge_jni.so usable instead of crashing the run.
     */
    private static double nativeMeanMs(int index) {
        try {
            long[] t = NativeOpenVinoBridge.nativeTimings();
            if (t == null || t.length < 3) {
                return 0.0;
            }
            long[] base = nativeBaseline;
            long calls = t[0] - base[0];
            if (calls <= 0) {
                return 0.0;
            }
            return ((t[index] - base[index]) / (double) calls) / 1_000_000.0;
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            return 0.0;
        }
    }

    private static String msMean(LatencyStats stats) {
        return formatDouble(stats == null ? 0.0 : stats.averageLatencyNs() / 1_000_000.0);
    }

    private static String msPercentile(LatencyStats stats, double percentile) {
        return formatDouble(stats == null ? 0.0 : stats.percentileLatencyNs(percentile) / 1_000_000.0);
    }

    private static String formatDouble(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    private static String csv(String value) {
        String safeValue = value == null ? "" : value;
        if (!safeValue.contains(",") && !safeValue.contains("\"") && !safeValue.contains("\n")) {
            return safeValue;
        }
        return "\"" + safeValue.replace("\"", "\"\"") + "\"";
    }
}
