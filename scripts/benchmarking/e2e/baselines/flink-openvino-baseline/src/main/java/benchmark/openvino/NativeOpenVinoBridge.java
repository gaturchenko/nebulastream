package benchmark.openvino;

public final class NativeOpenVinoBridge {
    static {
        System.loadLibrary("flink_openvino_bridge_jni");
    }

    private NativeOpenVinoBridge() {}

    public static native long createEngine(
            String modelPath,
            String device,
            int numStreams,
            int numThreads
    );

    public static native int inputSize(long handle);

    public static native int outputSize(long handle);

    public static native int predict(
            long handle,
            float[] input,
            float[] output
    );

    /**
     * Process-wide predict() timing totals: {calls, predictNs, inferNs}.
     *
     * predictNs spans the whole native predict (both tensor memcpys plus the inference), inferNs
     * only the inference. Subtracting them from the Java-side timing of this bridge call splits
     * the cost into JNI array access, native tensor marshalling, and model execution.
     */
    public static native long[] nativeTimings();

    public static native void destroyEngine(long handle);
}
