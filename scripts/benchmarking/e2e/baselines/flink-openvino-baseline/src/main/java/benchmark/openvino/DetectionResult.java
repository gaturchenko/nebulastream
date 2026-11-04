package benchmark.openvino;

import java.io.Serializable;

public final class DetectionResult implements Serializable {
    public final float maxScore;
    public final long e2eStartNs;
    /** Window sequence number for {@link WindowTrace}; -1 where nothing is traced. */
    public final long seq;

    public DetectionResult(float maxScore) {
        this(maxScore, 0L);
    }

    public DetectionResult(float maxScore, long e2eStartNs) {
        this(maxScore, e2eStartNs, -1L);
    }

    public DetectionResult(float maxScore, long e2eStartNs, long seq) {
        this.maxScore = maxScore;
        this.e2eStartNs = e2eStartNs;
        this.seq = seq;
    }

    @Override
    public String toString() {
        return "DetectionResult{" +
                "maxScore=" + maxScore +
                '}';
    }
}
