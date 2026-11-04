package benchmark.openvino;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class LatencyStats {
    private final List<Long> latenciesNs = new ArrayList<>();

    private long firstStartNs = Long.MAX_VALUE;
    private long lastEndNs = Long.MIN_VALUE;
    private long minLatencyNs = Long.MAX_VALUE;
    private long maxLatencyNs = Long.MIN_VALUE;
    private double totalLatencyNs;

    public void record(long startNs, long endNs) {
        long latencyNs = endNs - startNs;

        latenciesNs.add(latencyNs);
        firstStartNs = Math.min(firstStartNs, startNs);
        lastEndNs = Math.max(lastEndNs, endNs);
        minLatencyNs = Math.min(minLatencyNs, latencyNs);
        maxLatencyNs = Math.max(maxLatencyNs, latencyNs);
        totalLatencyNs += latencyNs;
    }

    public void merge(LatencyStats other) {
        if (other == null || other.latenciesNs.isEmpty()) {
            return;
        }

        latenciesNs.addAll(other.latenciesNs);
        firstStartNs = Math.min(firstStartNs, other.firstStartNs);
        lastEndNs = Math.max(lastEndNs, other.lastEndNs);
        minLatencyNs = Math.min(minLatencyNs, other.minLatencyNs);
        maxLatencyNs = Math.max(maxLatencyNs, other.maxLatencyNs);
        totalLatencyNs += other.totalLatencyNs;
    }

    public long count() {
        return latenciesNs.size();
    }

    public long elapsedNs() {
        if (latenciesNs.isEmpty()) {
            return 0L;
        }

        return lastEndNs - firstStartNs;
    }

    public double throughputRecordsPerSecond() {
        long elapsedNs = elapsedNs();
        if (elapsedNs <= 0L) {
            return 0.0;
        }

        return count() / (elapsedNs / 1_000_000_000.0);
    }

    public double averageLatencyNs() {
        if (latenciesNs.isEmpty()) {
            return 0.0;
        }

        return totalLatencyNs / latenciesNs.size();
    }

    public long minLatencyNs() {
        return latenciesNs.isEmpty() ? 0L : minLatencyNs;
    }

    public long maxLatencyNs() {
        return latenciesNs.isEmpty() ? 0L : maxLatencyNs;
    }

    public long percentileLatencyNs(double percentile) {
        if (latenciesNs.isEmpty()) {
            return 0L;
        }

        List<Long> sorted = new ArrayList<>(latenciesNs);
        Collections.sort(sorted);

        int index = (int) Math.ceil((percentile / 100.0) * sorted.size()) - 1;
        index = Math.max(0, Math.min(index, sorted.size() - 1));
        return sorted.get(index);
    }
}
