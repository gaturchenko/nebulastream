package benchmark.openvino;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public final class OpenVinoFlinkJob {
    public static void main(String[] args) throws Exception {
        String modelPath = args.length >= 1
                ? args[0]
                : "models/nanodet.xml";

        String device = args.length >= 2 ? args[1] : "CPU";

        long count = args.length >= 3
                ? Long.parseLong(args[2])
                : 1000L;

        int flinkParallelism = args.length >= 4
                ? Integer.parseInt(args[3])
                : 1;

        int openvinoStreams = args.length >= 5
                ? Integer.parseInt(args[4])
                : 1;

        int openvinoThreads = args.length >= 6
                ? Integer.parseInt(args[5])
                : 1;

        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment();

        int parallelism = args.length >= 4
                ? Integer.parseInt(args[3])
                : 1;

        env.setParallelism(parallelism);

        env
                .fromSequence(0, count - 1)
                .map(new OpenVinoRichMapFunction(
                        modelPath,
                        device,
                        openvinoStreams,
                        openvinoThreads
                ))
                .name("openvino-rich-map")
                .sinkTo(new DiscardSink<>());

        env.execute("Flink OpenVINO RichMapFunction Test");
    }
}
