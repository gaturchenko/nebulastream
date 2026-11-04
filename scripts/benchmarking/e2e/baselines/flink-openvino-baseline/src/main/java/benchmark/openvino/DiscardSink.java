package benchmark.openvino;

import org.apache.flink.api.connector.sink2.Sink;
import org.apache.flink.api.connector.sink2.SinkWriter;
import org.apache.flink.api.connector.sink2.WriterInitContext;

import java.io.IOException;

public final class DiscardSink<T> implements Sink<T> {

    @Override
    public SinkWriter<T> createWriter(WriterInitContext context) {
        return new SinkWriter<T>() {
            @Override
            public void write(T element, Context context) {
                // discard
            }

            @Override
            public void flush(boolean endOfInput) throws IOException {
                // no-op
            }

            @Override
            public void close() throws Exception {
                // no-op
            }
        };
    }
}
