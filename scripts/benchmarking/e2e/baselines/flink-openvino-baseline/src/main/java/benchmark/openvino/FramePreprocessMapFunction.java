package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.Base64;

public final class FramePreprocessMapFunction
        extends RichMapFunction<FrameRecord, PreprocessedFrameRecord> {

    private final int width;
    private final int height;
    /// base64 decode + JPEG decode + resize + tensor build, per frame. Reported so the comparison
    /// can say how much of Flink's end-to-end latency is per-frame work rather than inference.
    private transient LatencyStats stats;

    public FramePreprocessMapFunction(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public void open(OpenContext openContext) {
        stats = new LatencyStats();
    }

    @Override
    public void close() {
        if (stats != null) {
            StatsCsvWriter.record("preprocess", stats);
        }
    }

    @Override
    public PreprocessedFrameRecord map(FrameRecord frame) throws Exception {
        long startNs = System.nanoTime();
        float[] inputTensor = decodeAndPreprocess(frame.base64Frame);
        stats.record(startNs, System.nanoTime());
        return new PreprocessedFrameRecord(inputTensor, frame.e2eStartNs);
    }

    private float[] decodeAndPreprocess(String base64Frame) throws Exception {
        String cleanBase64 = stripDataUriPrefix(base64Frame);

        byte[] imageBytes = Base64.getDecoder().decode(cleanBase64);

        BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));

        if (image == null) {
            throw new IllegalArgumentException("Failed to decode image from base64 frame");
        }

        BufferedImage resized = resize(image, width, height);
        float[] dst = new float[3 * width * height];

        // NCHW layout: [1, 3, H, W], RGB channels scaled to [0, 1].
        int channelSize = width * height;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = resized.getRGB(x, y);

                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = rgb & 0xff;

                int offset = y * width + x;

                dst[offset] = r / 255.0f;
                dst[channelSize + offset] = g / 255.0f;
                dst[2 * channelSize + offset] = b / 255.0f;
            }
        }

        return dst;
    }

    private static String stripDataUriPrefix(String value) {
        int comma = value.indexOf(',');
        if (value.startsWith("data:") && comma >= 0) {
            return value.substring(comma + 1);
        }
        return value;
    }

    private static BufferedImage resize(BufferedImage src, int width, int height) {
        BufferedImage dst = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        Graphics2D g = dst.createGraphics();
        try {
            g.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );
            g.drawImage(src, 0, 0, width, height, null);
        } finally {
            g.dispose();
        }

        return dst;
    }
}
