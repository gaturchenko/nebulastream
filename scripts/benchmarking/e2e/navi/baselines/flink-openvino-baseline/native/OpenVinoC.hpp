#pragma once

#include <cstddef>

extern "C" {

void* ov_engine_create(
    const char* model_path,
    const char* device,
    int num_streams,
    int num_threads
);

void ov_engine_destroy(void* handle);

/// Process-wide predict() timing totals (ns), for the Java<->native transfer breakdown.
void ov_engine_timings(unsigned long long* calls, unsigned long long* predict_ns, unsigned long long* infer_ns);

int ov_engine_input_size(void* handle);

int ov_engine_output_size(void* handle);

int ov_engine_predict(
    void* handle,
    const float* input,
    int input_size,
    float* output,
    int output_size
);

}
