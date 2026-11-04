package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;

public final class OpenVinoFrameMapFunction
        extends RichMapFunction<PreprocessedFrameRecord, DetectionResult> {

    private final String modelPath;
    private final String device;
    private final int numStreams;
    private final int numThreads;
    private final String statsCsvPath;
    private final String runId;

    /**
     * Engine construction is serialised across the lanes of a JVM. Three concurrent
     * {@code compile_model} calls on a large IR (the 190 MB SARAD model) reproducibly wedge on a
     * 4-core/4 GB Pi: two lanes come up and the third never finishes, leaving the job parked
     * indefinitely. Compilation happens in open(), before the source is released by
     * {@link EngineReadiness}, so serialising it costs the measurement nothing -- it only flattens a
     * startup memory and CPU spike that has no bearing on steady-state serving.
     */
    private static final Object ENGINE_INIT_LOCK = new Object();

    private transient long handle;
    private transient float[] output;
    private transient int expectedInputSize;
    private transient LatencyStats stats;
    private transient int subtaskIndex;
    private transient int parallelSubtasks;

    public OpenVinoFrameMapFunction(
            String modelPath,
            String device,
            int numStreams,
            int numThreads,
            String statsCsvPath,
            String runId) {
        this.modelPath = modelPath;
        this.device = device;
        this.numStreams = numStreams;
        this.numThreads = numThreads;
        this.statsCsvPath = statsCsvPath;
        this.runId = runId;
    }

    @Override
    public void open(OpenContext openContext) {
        subtaskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
        parallelSubtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        stats = new LatencyStats();

        int inputSize;
        int outputSize;
        synchronized (ENGINE_INIT_LOCK) {
            handle = NativeOpenVinoBridge.createEngine(
                    modelPath,
                    device,
                    numStreams,
                    numThreads
            );

            inputSize = NativeOpenVinoBridge.inputSize(handle);
            outputSize = NativeOpenVinoBridge.outputSize(handle);

            expectedInputSize = inputSize;
            output = new float[outputSize];

            // One throwaway inference so the first measured call does not pay OpenVINO's lazy
            // per-request setup.
            NativeOpenVinoBridge.predict(handle, new float[inputSize], output);
        }

        // Report this lane ready: the paced source waits for all of them.
        EngineReadiness.ready();

        System.out.println(
                "Initialized OpenVINO frame engine. "
                        + "inputSize=" + inputSize
                        + ", outputSize=" + outputSize
                        + ", numStreams=" + numStreams
                        + ", numThreads=" + numThreads
        );
    }

    @Override
    public DetectionResult map(PreprocessedFrameRecord frame) {
        // Snapshot the native counters once, before the first measured call, so the warm-up
        // predicts every lane runs in open() do not land in the breakdown.
        StatsCsvWriter.markSteadyState();

        long startNs = System.nanoTime();

        if (frame.inputTensor.length != expectedInputSize) {
            throw new IllegalArgumentException(
                    "Unexpected preprocessed input size. Expected "
                            + expectedInputSize
                            + ", but got "
                            + frame.inputTensor.length
            );
        }

        int rc = NativeOpenVinoBridge.predict(handle, frame.inputTensor, output);

        if (rc != 0) {
            throw new RuntimeException("OpenVINO predict failed with rc=" + rc);
        }

        float maxScore = max(output);
        long endNs = System.nanoTime();
        stats.record(startNs, endNs);
        if (WindowTrace.enabled()) {
            WindowTrace.inferred(frame.seq, subtaskIndex, startNs, endNs);
        }

        return new DetectionResult(maxScore, frame.e2eStartNs, frame.seq);
    }

    @Override
    public void close() throws Exception {
        try {
            if (stats != null) {
                StatsCsvWriter.record("inference", stats);
            }
        } finally {
            if (handle != 0) {
                NativeOpenVinoBridge.destroyEngine(handle);
                handle = 0;
            }
        }
    }

    private static float max(float[] values) {
        float m = -Float.MAX_VALUE;

        for (float v : values) {
            if (v > m) {
                m = v;
            }
        }

        return m;
    }
}
