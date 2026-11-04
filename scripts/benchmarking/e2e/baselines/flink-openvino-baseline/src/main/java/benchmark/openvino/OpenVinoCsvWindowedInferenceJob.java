package benchmark.openvino;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

/**
 * Flink + OpenVINO baseline for the sensor-stream workloads (CWRU bearing vibration, MUSAN audio).
 *
 * It is the windowed counterpart of {@link OpenVinoCsvObjectDetectionJob}: instead of one inference
 * per frame, it aggregates a paced stream of scalar samples into sliding windows and runs one
 * inference per window, substituting the NES-UDF row in the framework comparison. Everything
 * downstream of the window -- the OpenVINO engine, its compile options and the stats schema -- is
 * shared with the object-detection job, so the two baselines report identical columns.
 *
 * <p>Arguments (positional; the first two are required):
 *
 * <pre>
 *   0  modelPath        OpenVINO IR (.xml) or .onnx; the engine pins a dynamic batch dim to 1
 *   1  csvPath          sample CSV, one sample per row
 *   2  device           CPU
 *   3  flinkParallelism inference lanes, matched to the NES worker thread count
 *   4  ovStreams        OpenVINO num_streams   (negative leaves it unset; 0 is explicit, like NES)
 *   5  ovThreads        OpenVINO inference_num_threads
 *   6  valueColumn      0-indexed column holding the sample value
 *   7  hasHeader        skip the first row
 *   8  windowSamples    samples per model input  (SIZE    in samples)
 *   9  advanceSamples   samples between windows  (ADVANCE in samples)
 *  10  scaleDivisor     divide each sample by this (<= 0 to leave raw), e.g. 32767 for int16 audio
 *  11  printResults     print instead of discarding the model output
 *  12  statsCsvPath     comparison row destination
 *  13  queryName        comparison-key columns, as in the object-detection job
 *  14  paramName
 *  15  paramValue
 *  16  repetition
 *  17  ratePerSecond    ingestion pacing, matched to the NES source MOCK_TUPLE_RATE
 * </pre>
 */
public final class OpenVinoCsvWindowedInferenceJob {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println(
                    "Usage: OpenVinoCsvWindowedInferenceJob "
                            + "<modelPath> <csvPath> "
                            + "[device] [flinkParallelism] [ovStreams] [ovThreads] "
                            + "[valueColumn] [hasHeader] [windowSamples] [advanceSamples] "
                            + "[scaleDivisor] [printResults] [statsCsvPath] "
                            + "[queryName] [paramName] [paramValue] [repetition] [ratePerSecond]"
            );
            System.exit(1);
        }

        String modelPath = args[0];
        String csvPath = args[1];

        String device = args.length >= 3 ? args[2] : "CPU";
        int flinkParallelism = args.length >= 4 ? Integer.parseInt(args[3]) : 1;
        int ovStreams = args.length >= 5 ? Integer.parseInt(args[4]) : 1;
        int ovThreads = args.length >= 6 ? Integer.parseInt(args[5]) : 1;

        int valueColumn = args.length >= 7 ? Integer.parseInt(args[6]) : 1;
        boolean hasHeader = args.length >= 8 && Boolean.parseBoolean(args[7]);

        int windowSamples = args.length >= 9 ? Integer.parseInt(args[8]) : 9600;
        int advanceSamples = args.length >= 10 ? Integer.parseInt(args[9]) : 480;
        float scaleDivisor = args.length >= 11 ? Float.parseFloat(args[10]) : 0.0f;

        boolean printResults = args.length >= 12 && Boolean.parseBoolean(args[11]);
        String statsCsvPath = args.length >= 13 ? args[12] : "target/openvino-windowed-stats.csv";
        String runId = "run-" + System.currentTimeMillis();

        String queryName = args.length >= 14 ? args[13] : "flink_windowed";
        String paramName = args.length >= 15 ? args[14] : "baseline";
        String paramValue = args.length >= 16 ? args[15] : "flink";
        String repetition = args.length >= 17 ? args[16] : "0";
        StatsCsvWriter.configureRun(queryName, paramName, paramValue, repetition);

        double ratePerSecond = args.length >= 18 ? Double.parseDouble(args[17]) : 0.0;

        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(flinkParallelism);

        // Flush output buffers as soon as a record is written, instead of letting them wait for
        // Flink's default buffer timeout (100 ms in 2.x). The only shuffle here is the
        // parallelism-1 parse -> parallelism-N inference boundary, and a record that does not fill
        // the last 32 KB network buffer sits there until the timeout expires or the next record
        // arrives -- at a paced 10 records/s, that is up to 100 ms of pure waiting charged to
        // latency. Leaving it at the default measured Flink's batching heuristic rather than its
        // serving path, and nobody tuning this job for latency would leave it unset.
        env.setBufferTimeout(0);

        FileSource<String> source = FileSource
                .forRecordStreamFormat(
                        new TextLineInputFormat(),
                        new Path(csvPath)
                )
                .build();

        // The windowing stage runs single-threaded: the samples of one window have to be ordered and
        // contiguous, which a parallel reader over one file would break, and it is also where the
        // single paced source lives. Inference then fans out over flinkParallelism lanes, the same
        // shape as the object-detection job (paced parse at 1, model at N).
        var results = env
                .fromSource(
                        source,
                        WatermarkStrategy.noWatermarks(),
                        "csv-sample-source"
                )
                .flatMap(new SlidingWindowSampleParser(
                        valueColumn,
                        hasHeader,
                        ratePerSecond,
                        windowSamples,
                        advanceSamples,
                        scaleDivisor,
                        flinkParallelism
                ))
                .name("sliding-window-aggregate")
                .setParallelism(1)
                .rebalance()
                .map(new OpenVinoFrameMapFunction(
                        modelPath,
                        device,
                        ovStreams,
                        ovThreads,
                        statsCsvPath,
                        runId
                ))
                .name("openvino-inference")
                .map(new E2eStatsMapFunction(statsCsvPath, runId))
                .name("collect-e2e-stats");

        if (printResults) {
            results.print();
        } else {
            results.sinkTo(new DiscardSink<>());
        }

        env.execute("Flink OpenVINO CSV Windowed Inference");

        StatsCsvWriter.writeWideRow(statsCsvPath);
        WindowTrace.write();
    }
}
