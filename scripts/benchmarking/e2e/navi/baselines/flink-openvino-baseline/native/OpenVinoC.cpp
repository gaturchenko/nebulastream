#include "OpenVinoC.hpp"

#include <cstdint>
#include "OpenVinoEngine.hpp"

#include <exception>
#include <memory>

extern "C" {

    void* ov_engine_create(
        const char* model_path,
        const char* device,
        int num_streams,
        int num_threads
    ) {
        try {
            return new OpenVinoEngine(
                model_path,
                device,
                num_streams,
                num_threads
            );
        } catch (const std::exception& e) {
            std::cerr << "ov_engine_create failed: " << e.what() << "\n";
            return nullptr;
        } catch (...) {
            std::cerr << "ov_engine_create failed: unknown exception\n";
            return nullptr;
        }
    }

    void ov_engine_destroy(void* handle) {
        delete static_cast<OpenVinoEngine*>(handle);
    }

    int ov_engine_input_size(void* handle) {
        if (!handle) return -1;
        return static_cast<int>(
            static_cast<OpenVinoEngine*>(handle)->input_size()
        );
    }

    int ov_engine_output_size(void* handle) {
        if (!handle) return -1;
        return static_cast<int>(
            static_cast<OpenVinoEngine*>(handle)->output_size()
        );
    }

    int ov_engine_predict(
        void* handle,
        const float* input,
        int input_size,
        float* output,
        int output_size
    ) {
        if (!handle || !input || !output) {
            return -1;
        }

        auto* engine = static_cast<OpenVinoEngine*>(handle);

        if (input_size != static_cast<int>(engine->input_size())) {
            return -2;
        }

        if (output_size != static_cast<int>(engine->output_size())) {
            return -3;
        }

        try {
            engine->predict(input, output);
            return 0;
        } catch (...) {
            return -4;
        }
    }

}

void ov_engine_timings(unsigned long long* calls, unsigned long long* predict_ns, unsigned long long* infer_ns) {
    std::uint64_t c = 0;
    std::uint64_t p = 0;
    std::uint64_t i = 0;
    OpenVinoEngine::timings(c, p, i);
    if (calls != nullptr) { *calls = c; }
    if (predict_ns != nullptr) { *predict_ns = p; }
    if (infer_ns != nullptr) { *infer_ns = i; }
}
