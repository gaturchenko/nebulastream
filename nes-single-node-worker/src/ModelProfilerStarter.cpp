/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

/// Runs the profiling job that the inference optimizer submits to a worker, outside of a query, and prints the per-batch-size
/// invocation latencies as CSV. The numbers come from the same code path as the optimizer's (profileInferenceModel with the
/// worker's own configuration), so the post-join decision k * t(1) vs. t(k) can be predicted for any k without deploying a query.
///
///   nes-model-profiler --model=/path/model.onnx --batch_sizes=1,2,3,4,6,8 --repetitions=5 \
///       --worker.default_query_execution.inference.openvino_inference_num_threads=4 > profile.csv

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Configuration/WorkerConfiguration.hpp>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/Util.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <cpptrace/from_current.hpp>
#include <ErrorHandling.hpp>
#include <Inference.hpp>
#include <InferenceProfiling.hpp>
#include <ModelProfile.hpp>

namespace NES
{

class ModelProfilerConfiguration final : public BaseConfiguration
{
public:
    StringOption modelPath = {"model", "", "Model to profile: an ONNX file or a TensorFlow SavedModel directory."};
    StringOption label = {"label", "", "Name of the model in the CSV output. Defaults to the model's file name."};
    StringOption batchSizes = {"batch_sizes", "1,2,3,4,6,8", "Comma-separated batch sizes to profile."};
    /// The defaults mirror the optimizer's inference_profiling_warmup_iterations and inference_profiling_iterations.
    UIntOption warmupIterations = {"warmup_iterations", "10", "Unmeasured model invocations per batch size."};
    UIntOption measurementIterations = {"iterations", "50", "Measured model invocations per batch size."};
    UIntOption repetitions = {"repetitions", "5", "Profiling jobs to run, each with the batch sizes in a different order."};
    WorkerConfiguration workerConfiguration = {"worker", "Worker configuration the profiled sessions are set up with"};

protected:
    std::vector<BaseOption*> getOptions() override
    {
        return {&modelPath, &label, &batchSizes, &warmupIterations, &measurementIterations, &repetitions, &workerConfiguration};
    }
};

namespace
{

std::vector<size_t> parseBatchSizes(const std::string& list)
{
    std::vector<size_t> batchSizes;
    std::stringstream stream{list};
    std::string item;
    while (std::getline(stream, item, ','))
    {
        const auto batchSize = std::stoull(item);
        if (batchSize == 0)
        {
            throw InvalidConfigParameter("batch_sizes must be positive, got '{}'", list);
        }
        batchSizes.push_back(batchSize);
    }
    if (batchSizes.empty())
    {
        throw InvalidConfigParameter("batch_sizes is empty");
    }
    return batchSizes;
}

}
}

int main(const int argc, const char* argv[])
{
    CPPTRACE_TRY
    {
        /// stdout carries the CSV, so the log only goes to its file.
        NES::Logger::setupLogging("modelProfiler.log", NES::LogLevel::LOG_INFO, false);
        auto configuration = NES::loadConfiguration<NES::ModelProfilerConfiguration>(argc, argv);
        if (!configuration)
        {
            return 0;
        }

        const std::filesystem::path modelPath = configuration->modelPath.getValue();
        if (modelPath.empty())
        {
            std::cerr << "--model is required\n";
            return 1;
        }
        auto batchSizes = NES::parseBatchSizes(configuration->batchSizes.getValue());
        const auto label = configuration->label.getValue().empty() ? modelPath.filename().string() : configuration->label.getValue();

        const auto imported = NES::importModel(modelPath);
        if (!imported)
        {
            std::cerr << "Failed to import " << modelPath << ": " << imported.error().message << '\n';
            return 1;
        }

        const auto& worker = configuration->workerConfiguration;
        const auto& queryExecution = worker.defaultQueryExecution;
        const auto workerThreads = worker.queryEngine.numberOfWorkerThreads.getValue();
        /// The options the sessions are actually set up with: sharing the compiled model raises streams and threads.
        const auto options = NES::resolveSessionOptions(NES::getInferenceRuntimeOptions(queryExecution), workerThreads);
        const auto dynamicBatch = NES::usesDynamicInferenceBatches(queryExecution);

        std::cout << "model,worker_threads,openvino_inference_num_threads,openvino_num_streams,share_compiled_model,dynamic_batch,"
                     "repetition,batch_size,iterations,median_ns,mean_ns,min_ns\n";
        for (uint64_t repetition = 1; repetition <= configuration->repetitions.getValue(); ++repetition)
        {
            /// Profiling the batch sizes in a fixed order would bias whichever runs last on a warming SoC.
            std::mt19937 generator{static_cast<std::mt19937::result_type>(repetition)};
            std::ranges::shuffle(batchSizes, generator);
            const auto profile = NES::profileInferenceModel(
                NES::ModelProfilingRequest{
                    .model = *imported,
                    .batchSizes = batchSizes,
                    .warmupIterations = configuration->warmupIterations.getValue(),
                    .measurementIterations = configuration->measurementIterations.getValue(),
                    .assessMemoryForBatchSize = std::nullopt},
                worker);
            for (const auto& batch : profile.batches)
            {
                std::cout << label << ',' << workerThreads << ',' << options.openvinoInferenceNumThreads << ','
                          << options.openvinoNumStreams << ',' << options.openvinoShareCompiledModel << ',' << dynamicBatch << ','
                          << repetition << ',' << batch.batchSize << ',' << batch.iterations << ',' << batch.medianLatency.count()
                          << ',' << batch.meanLatency.count() << ',' << batch.minLatency.count() << '\n';
            }
            std::cout.flush();
        }
        NES::Logger::getInstance()->forceFlush();
        return 0;
    }
    CPPTRACE_CATCH(...)
    {
        NES::tryLogCurrentException();
        return NES::getCurrentErrorCode();
    }
}
