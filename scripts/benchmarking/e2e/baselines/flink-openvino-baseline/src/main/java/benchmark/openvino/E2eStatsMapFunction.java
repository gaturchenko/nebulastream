package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;

public final class E2eStatsMapFunction
        extends RichMapFunction<DetectionResult, DetectionResult> {

    private final String statsCsvPath;
    private final String runId;

    private transient LatencyStats stats;
    private transient int subtaskIndex;
    private transient int parallelSubtasks;

    public E2eStatsMapFunction(String statsCsvPath, String runId) {
        this.statsCsvPath = statsCsvPath;
        this.runId = runId;
    }

    @Override
    public void open(OpenContext openContext) {
        subtaskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
        parallelSubtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        stats = new LatencyStats();
    }

    @Override
    public DetectionResult map(DetectionResult result) {
        long endNs = System.nanoTime();
        stats.record(result.e2eStartNs, endNs);
        if (WindowTrace.enabled()) {
            WindowTrace.delivered(result.seq, result.e2eStartNs, endNs);
        }
        return result;
    }

    @Override
    public void close() throws Exception {
        if (stats != null) {
            StatsCsvWriter.record("e2e", stats);
        }
    }
}
