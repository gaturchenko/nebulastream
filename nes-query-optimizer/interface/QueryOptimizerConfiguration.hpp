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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/Enums/EnumOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/Validation/NonZeroValidation.hpp>
#include <QueryOptimizerNetworkConfiguration.hpp>

namespace NES
{

enum class StreamJoinStrategy : uint8_t
{
    NESTED_LOOP_JOIN,
    HASH_JOIN,
    OPTIMIZER_CHOOSES
};

/// How the batch size of inference operators is chosen.
enum class InferenceBatchSizeStrategy : uint8_t
{
    /// Use the configured inference_batch_size for every inference operator.
    FIXED,
    /// Submit a profiling job to the worker hosting an inference operator fed by a time window and choose the batch size with the
    /// highest throughput among those that keep up with the window results and meet the latency SLO derived from the window. A
    /// batch size keeps up if its invocation occupies at most inference_max_utilization of the B strides in which a batch of B
    /// results arrives; it meets the SLO if the batch fill time plus the invocation stays within the window size. Other inference
    /// operators use the configured inference_batch_size.
    OPTIMIZER_CHOOSES
};

/// How the memory-related deployment of inference operators is chosen: whether worker threads share one compiled model, and
/// whether they share one prediction cache.
enum class InferenceMemoryStrategy : uint8_t
{
    /// Use the worker's configured openvino_share_compiled_model and prediction_cache_scope.
    FIXED,
    /// Compare what the alternatives cost against the memory the worker has, and pick the fastest one that fits.
    OPTIMIZER_CHOOSES
};

/// How an inference operator over k compatible payload fields of a joined record invokes the model.
enum class PostJoinInferenceStrategy : uint8_t
{
    /// Always invoke the model once with a batch of k payloads.
    BATCHED,
    /// Always invoke the model k times with a single payload.
    SEQUENTIAL,
    /// Submit a profiling job to the worker hosting the operator and batch only if a batch-k invocation amortizes over k
    /// single-tuple invocations.
    OPTIMIZER_CHOOSES
};

class QueryOptimizerConfiguration : public BaseConfiguration
{
public:
    QueryOptimizerConfiguration() = default;
    QueryOptimizerConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    EnumOption<StreamJoinStrategy> joinStrategy
        = {"join_strategy",
           StreamJoinStrategy::OPTIMIZER_CHOOSES,
           "Join Strategy"
           "[NESTED_LOOP_JOIN|HASH_JOIN|OPTIMIZER_CHOOSES]."};

    QueryOptimizerNetworkConfiguration network = {"network", "Network configuration overrides for query decomposition"};

    UIntOption inferenceBatchSize = {"inference_batch_size", "1", "Inference Batch Size", {std::make_shared<NonZeroValidation>()}};

    EnumOption<InferenceBatchSizeStrategy> inferenceBatchSizeStrategy
        = {"inference_batch_size_strategy",
           InferenceBatchSizeStrategy::FIXED,
           "How the batch size of inference operators is chosen [FIXED|OPTIMIZER_CHOOSES]. OPTIMIZER_CHOOSES profiles candidate "
           "batch sizes on the worker hosting an inference operator fed by a time window and picks the highest-throughput batch size "
           "that keeps up with the window results and meets the window's latency SLO."};

    UIntOption inferenceMaxBatchSize
        = {"inference_max_batch_size",
           "256",
           "Largest batch size OPTIMIZER_CHOOSES considers. Candidates are the powers of two up to this size.",
           {std::make_shared<NonZeroValidation>()}};

    FloatOption inferenceReplaySpeedup
        = {"inference_replay_speedup",
           "1.0",
           "Factor by which the sources replay event time faster than wall-clock time. OPTIMIZER_CHOOSES divides the window stride "
           "and size by it to obtain the wall-clock arrival interval and latency SLO of window results."};

    FloatOption inferenceMaxUtilization
        = {"inference_max_utilization",
           "0.8",
           "Largest fraction of the time between two batches that OPTIMIZER_CHOOSES lets a batched invocation occupy. A batch size "
           "above it does not keep up with the window results: the headroom absorbs the queueing that makes latency diverge as the "
           "utilization approaches 1."};

    EnumOption<InferenceMemoryStrategy> inferenceMemoryStrategy
        = {"inference_memory_strategy",
           InferenceMemoryStrategy::FIXED,
           "How compiled-model sharing and the prediction cache scope are chosen [FIXED|OPTIMIZER_CHOOSES]. OPTIMIZER_CHOOSES has the "
           "worker assess what an inference operator costs in memory and picks the fastest deployment that fits its memory budget."};

    FloatOption inferenceMemoryBudgetFraction
        = {"inference_memory_budget_fraction",
           "0.8",
           "Fraction of the worker's memory that is left after its buffer pool that inference operators may occupy under "
           "OPTIMIZER_CHOOSES."};

    EnumOption<PostJoinInferenceStrategy> postJoinInferenceStrategy
        = {"post_join_inference_strategy",
           PostJoinInferenceStrategy::OPTIMIZER_CHOOSES,
           "How inference over k compatible payload fields of a joined record invokes the model "
           "[BATCHED|SEQUENTIAL|OPTIMIZER_CHOOSES]. OPTIMIZER_CHOOSES profiles one batch-k invocation against k single-tuple "
           "invocations on the worker hosting the operator."};

    FloatOption postJoinInferenceMinimumSpeedup
        = {"post_join_inference_minimum_speedup",
           "1.1",
           "Minimum speedup of one batch-k invocation over k single-tuple invocations for which OPTIMIZER_CHOOSES selects "
           "post-join batching. The margin covers the batching overhead that is not part of the profiled model invocation."};

    UIntOption inferenceProfilingWarmupIterations
        = {"inference_profiling_warmup_iterations", "10", "Unmeasured model invocations per batch size before profiling."};

    UIntOption inferenceProfilingIterations
        = {"inference_profiling_iterations",
           "50",
           "Measured model invocations per batch size when profiling.",
           {std::make_shared<NonZeroValidation>()}};

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {
            &joinStrategy,
            &network,
            &inferenceBatchSize,
            &inferenceBatchSizeStrategy,
            &inferenceMaxBatchSize,
            &inferenceReplaySpeedup,
            &inferenceMaxUtilization,
            &inferenceMemoryStrategy,
            &inferenceMemoryBudgetFraction,
            &postJoinInferenceStrategy,
            &postJoinInferenceMinimumSpeedup,
            &inferenceProfilingWarmupIterations,
            &inferenceProfilingIterations};
    }
};

}
