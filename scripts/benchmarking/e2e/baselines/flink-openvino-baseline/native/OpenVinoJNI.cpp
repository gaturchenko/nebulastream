#include "OpenVinoC.hpp"

#include <jni.h>
#include <string>

static std::string jstring_to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static void throw_runtime_exception(JNIEnv* env, const char* message) {
    jclass exception_class = env->FindClass("java/lang/RuntimeException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_createEngine(
        JNIEnv* env,
        jclass,
        jstring modelPath,
        jstring device,
        jint numStreams,
        jint numThreads) {

    const std::string model_path = jstring_to_string(env, modelPath);
    const std::string device_name = jstring_to_string(env, device);

    void* handle = ov_engine_create(
        model_path.c_str(),
        device_name.c_str(),
        static_cast<int>(numStreams),
        static_cast<int>(numThreads)
    );

    if (handle == nullptr) {
        throw_runtime_exception(env, "Failed to create OpenVINO engine");
        return 0;
    }

    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT jint JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_inputSize(
        JNIEnv* env,
        jclass,
        jlong handle) {

    if (handle == 0) {
        throw_runtime_exception(env, "OpenVINO engine handle is null");
        return -1;
    }

    return ov_engine_input_size(reinterpret_cast<void*>(handle));
}

JNIEXPORT jint JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_outputSize(
        JNIEnv* env,
        jclass,
        jlong handle) {

    if (handle == 0) {
        throw_runtime_exception(env, "OpenVINO engine handle is null");
        return -1;
    }

    return ov_engine_output_size(reinterpret_cast<void*>(handle));
}

JNIEXPORT jint JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_predict(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloatArray inputArray,
        jfloatArray outputArray) {

    if (handle == 0) {
        throw_runtime_exception(env, "OpenVINO engine handle is null");
        return -1;
    }

    if (inputArray == nullptr || outputArray == nullptr) {
        throw_runtime_exception(env, "Input/output array is null");
        return -1;
    }

    const jsize input_size = env->GetArrayLength(inputArray);
    const jsize output_size = env->GetArrayLength(outputArray);

    jfloat* input = env->GetFloatArrayElements(inputArray, nullptr);
    if (input == nullptr) {
        throw_runtime_exception(env, "Failed to access input array");
        return -1;
    }

    jfloat* output = env->GetFloatArrayElements(outputArray, nullptr);
    if (output == nullptr) {
        env->ReleaseFloatArrayElements(inputArray, input, JNI_ABORT);
        throw_runtime_exception(env, "Failed to access output array");
        return -1;
    }

    const int rc = ov_engine_predict(
            reinterpret_cast<void*>(handle),
            input,
            static_cast<int>(input_size),
            output,
            static_cast<int>(output_size)
    );

    env->ReleaseFloatArrayElements(inputArray, input, JNI_ABORT);
    env->ReleaseFloatArrayElements(outputArray, output, 0);

    return rc;
}

JNIEXPORT jlongArray JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_nativeTimings(
        JNIEnv* env,
        jclass) {

    unsigned long long calls = 0;
    unsigned long long predict_ns = 0;
    unsigned long long infer_ns = 0;
    ov_engine_timings(&calls, &predict_ns, &infer_ns);

    jlongArray result = env->NewLongArray(3);
    if (result == nullptr) {
        return nullptr;
    }
    jlong values[3] = {
        static_cast<jlong>(calls),
        static_cast<jlong>(predict_ns),
        static_cast<jlong>(infer_ns)
    };
    env->SetLongArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT void JNICALL
Java_benchmark_openvino_NativeOpenVinoBridge_destroyEngine(
        JNIEnv*,
        jclass,
        jlong handle) {

    if (handle != 0) {
        ov_engine_destroy(reinterpret_cast<void*>(handle));
    }
}

}
