package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;

public final class OpenVinoRichMapFunction
        extends RichMapFunction<Long, Float> {

    private final String modelPath;
    private final String device;
    private final int numStreams;
    private final int numThreads;

    private transient long handle;
    private transient float[] input;
    private transient float[] output;

    private transient long seen;
    private transient long totalLatencyNs;
    private transient long startWallNs;

    public OpenVinoRichMapFunction(
            String modelPath,
            String device,
            int numStreams,
            int numThreads) {
        this.modelPath = modelPath;
        this.device = device;
        this.numStreams = numStreams;
        this.numThreads = numThreads;
    }

    @Override
    public void open(OpenContext openContext) {
        seen = 0;
        totalLatencyNs = 0;
        startWallNs = System.nanoTime();

        handle = NativeOpenVinoBridge.createEngine(
                modelPath,
                device,
                numStreams,
                numThreads
        );

        int inputSize = NativeOpenVinoBridge.inputSize(handle);
        int outputSize = NativeOpenVinoBridge.outputSize(handle);

        input = new float[inputSize];
        output = new float[outputSize];

        for (int i = 0; i < input.length; i++) {
            input[i] = ((i % 127) - 63) / 64.0f;
        }

        System.out.println(
                "Initialized OpenVINO engine. inputSize=" + inputSize
                        + ", outputSize=" + outputSize
                        + ", numStreams=" + numStreams
                        + ", numThreads=" + numThreads
        );
    }

    @Override
    public Float map(Long value) {
        long start = System.nanoTime();

        int rc = NativeOpenVinoBridge.predict(handle, input, output);

        long latencyNs = System.nanoTime() - start;

        if (rc != 0) {
            throw new RuntimeException("OpenVINO inference failed with rc=" + rc);
        }

        seen++;
        totalLatencyNs += latencyNs;

        if (seen % 100 == 0) {
            double elapsedSec = (System.nanoTime() - startWallNs) / 1_000_000_000.0;
            double throughput = seen / elapsedSec;
            double avgLatencyUs = (totalLatencyNs / (double) seen) / 1000.0;

            System.out.printf(
                    "records=%d throughput=%.2f rec/s avg_inference=%.3f us%n",
                    seen,
                    throughput,
                    avgLatencyUs
            );
        }

        return output[0];
    }

    @Override
    public void close() {
        if (handle != 0L) {
            NativeOpenVinoBridge.destroyEngine(handle);
            handle = 0L;
        }
    }
}
