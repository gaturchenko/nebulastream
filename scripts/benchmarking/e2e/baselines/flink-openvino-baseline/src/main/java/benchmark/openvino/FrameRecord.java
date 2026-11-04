package benchmark.openvino;

import java.io.Serializable;

public final class FrameRecord implements Serializable {
    public final String base64Frame;
    public final long e2eStartNs;

    public FrameRecord(String base64Frame) {
        this(base64Frame, System.nanoTime());
    }

    public FrameRecord(String base64Frame, long e2eStartNs) {
        this.base64Frame = base64Frame;
        this.e2eStartNs = e2eStartNs;
    }
}
