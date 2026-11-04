package benchmark.openvino;

import java.io.Serializable;
import java.util.Objects;

public final class PreprocessedFrameRecord implements Serializable {
    public final float[] inputTensor;
    public final long e2eStartNs;
    /** Window sequence number for {@link WindowTrace}; -1 where nothing is traced. */
    public final long seq;

    public PreprocessedFrameRecord(float[] inputTensor, long e2eStartNs) {
        this(inputTensor, e2eStartNs, -1L);
    }

    public PreprocessedFrameRecord(float[] inputTensor, long e2eStartNs, long seq) {
        this.inputTensor = Objects.requireNonNull(inputTensor, "inputTensor");
        this.e2eStartNs = e2eStartNs;
        this.seq = seq;
    }
}
