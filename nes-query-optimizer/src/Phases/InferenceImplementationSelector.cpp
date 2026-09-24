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

#include <Phases/InferenceImplementationSelector.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/ProjectionLogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/InferenceBatchSizeTrait.hpp>
#include <Traits/PlacementTrait.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/Logger.hpp>
#include <WindowTypes/Measures/TimeMeasure.hpp>
#include <WindowTypes/Types/TimeBasedWindowType.hpp>
#include <magic_enum/magic_enum.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
#include <ModelCatalog.hpp>
#include <ModelProfile.hpp>
#include <QueryOptimizerConfiguration.hpp>

namespace NES
{

namespace
{

/// Mirrors the lowering, which batches an inference operator's payload fields when the model consumes a single VARSIZED
/// input and the operator reads it from several fields of a joined record.
bool isPostJoinInference(const InferModelLogicalOperator& inferModel)
{
    return inferModel.hasVarsizedInput() && inferModel.getInputFieldNames().size() > 1;
}

/// Finds the windowed aggregation whose results reach the inference operator one-to-one. Only sequence and projection operators may
/// lie in between; any other operator, e.g., a selection, changes how many results arrive per window stride.
std::optional<WindowedAggregationLogicalOperator> findFeedingWindow(const LogicalOperator& inferModel)
{
    auto children = inferModel.getChildren();
    while (children.size() == 1)
    {
        const auto& child = children.front();
        if (const auto window = child.tryGetAs<WindowedAggregationLogicalOperator>(); window.has_value())
        {
            return window.value().get();
        }
        if (!child.tryGetAs<SequenceLogicalOperator>().has_value() && !child.tryGetAs<ProjectionLogicalOperator>().has_value())
        {
            return std::nullopt;
        }
        children = child.getChildren();
    }
    return std::nullopt;
}

std::vector<LogicalOperator> withoutInferenceSequence(std::vector<LogicalOperator> children)
{
    if (children.size() != 1)
    {
        return children;
    }
    const auto sequence = children.front().tryGetAs<SequenceLogicalOperator>();
    if (!sequence.has_value() || sequence.value().get().getSequenceSource() != SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        return children;
    }
    return children.front().getChildren();
}

/// Reports what the model costs on the worker, so that the memory a deployment needs can be compared against what the worker has.
void logMemoryProfile(const Host& worker, const std::string& modelName, const InferenceMemoryProfile& memory)
{
    const auto threads = memory.worker.numberOfWorkerThreads;
    if (!memory.session.measured)
    {
        NES_WARNING("No memory assessment for model '{}' on worker '{}'", modelName, worker);
        return;
    }
    NES_INFO(
        "Inference memory for model '{}' on worker '{}': first session {} B, each additional session {} B private or {} B sharing "
        "the compiled model; {} sessions need {} B with a private compiled model each or {} B sharing one; a prediction cache of {} "
        "entries needs {} B thread-local or {} B global; the worker has {} B physical memory, {} B of which its buffer pool "
        "reserves, leaving {} B",
        modelName,
        worker,
        memory.session.firstPrivateSessionBytes,
        memory.session.additionalPrivateSessionBytes,
        memory.session.additionalSharedSessionBytes,
        threads,
        memory.session.privateDeploymentBytes(threads),
        memory.session.sharedDeploymentBytes(threads),
        memory.predictionCache.entries,
        memory.predictionCache.threadLocalBytes,
        memory.predictionCache.globalBytes,
        memory.worker.physicalMemoryBytes,
        memory.worker.bufferPoolBytes,
        memory.worker.availableBytes());
}

struct BatchSizeCandidate
{
    size_t batchSize;
    std::chrono::nanoseconds invocationLatency;
    std::chrono::nanoseconds resultLatency;
    double throughput;
    bool meetsHardSlo;
    bool meetsWeakSlo;
};

}

InferenceImplementationSelector::InferenceImplementationSelector(
    QueryOptimizerConfiguration configuration, std::shared_ptr<InferenceProfiler> profiler)
    : configuration(std::move(configuration)), profiler(std::move(profiler)), profileCache(std::make_shared<ProfileCache>())
{
}

LogicalPlan InferenceImplementationSelector::apply(const LogicalPlan& plan) const
{
    return plan.withRootOperators(
        plan.getRootOperators() | std::views::transform([this](const LogicalOperator& root) { return apply(root); })
        | std::ranges::to<std::vector>());
}

LogicalOperator InferenceImplementationSelector::apply(const LogicalOperator& logicalOperator) const
{
    auto children = logicalOperator.getChildren() | std::views::transform([this](const LogicalOperator& child) { return apply(child); })
        | std::ranges::to<std::vector>();
    auto traitSet = logicalOperator.getTraitSet();
    if (const auto inferModel = logicalOperator.tryGetAs<InferModelLogicalOperator>(); inferModel.has_value())
    {
        bool batched = false;
        if (isPostJoinInference(inferModel.value().get()))
        {
            const auto implementation = selectPostJoinImplementation(logicalOperator, inferModel.value().get());
            tryInsert(traitSet, PostJoinInferenceImplementationTrait{implementation});
            batched = implementation == PostJoinInferenceImplementation::BATCHED;
        }
        else if (configuration.inferenceBatchSizeStrategy.getValue() == InferenceBatchSizeStrategy::OPTIMIZER_CHOOSES)
        {
            const auto batchSize = selectBatchSize(logicalOperator, inferModel.value().get());
            tryInsert(traitSet, InferenceBatchSizeTrait{batchSize});
            batched = batchSize > 1;
        }
        else
        {
            batched = configuration.inferenceBatchSize.getValue() > 1;
        }

        /// The inference sequence operator restores the record order that batched inference needs. It was inserted before the batch
        /// size was known, and single-tuple inference neither needs it nor can be lowered into its pipeline.
        if (!batched)
        {
            children = withoutInferenceSequence(std::move(children));
        }
    }
    return logicalOperator.withChildren(std::move(children)).withTraitSet(traitSet);
}

PostJoinInferenceImplementation InferenceImplementationSelector::selectPostJoinImplementation(
    const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const
{
    switch (configuration.postJoinInferenceStrategy.getValue())
    {
        case PostJoinInferenceStrategy::BATCHED:
            return PostJoinInferenceImplementation::BATCHED;
        case PostJoinInferenceStrategy::SEQUENTIAL:
            return PostJoinInferenceImplementation::SEQUENTIAL;
        case PostJoinInferenceStrategy::OPTIMIZER_CHOOSES:
            break;
    }

    const auto& modelName = inferModel.getModel().getName();
    const auto numberOfPayloads = inferModel.getInputFieldNames().size();
    const auto placement = getTrait<PlacementTrait>(logicalOperator.getTraitSet());
    if (!profiler || !placement.has_value())
    {
        NES_WARNING(
            "Cannot profile post-join inference of model '{}' ({}), falling back to post-join batching",
            modelName,
            profiler ? "operator is not placed" : "no inference profiler is available");
        return PostJoinInferenceImplementation::BATCHED;
    }

    const auto& worker = placement.value()->onNode;
    const auto profile = getProfile(worker, inferModel, {1, numberOfPayloads});
    const auto singleTuple = profile.has_value() ? profile->forBatchSize(1) : std::nullopt;
    const auto batched = profile.has_value() ? profile->forBatchSize(numberOfPayloads) : std::nullopt;
    if (!singleTuple.has_value() || !batched.has_value())
    {
        NES_WARNING(
            "Profiling model '{}' on worker '{}' failed ({}), falling back to post-join batching",
            modelName,
            worker,
            profile.has_value() ? "incomplete profile" : profile.error().what());
        return PostJoinInferenceImplementation::BATCHED;
    }

    /// k single-tuple invocations are compared against one batch-k invocation. The batch amortizes if its speedup reaches the
    /// configured minimum, which accounts for the batching overhead outside the profiled model invocation.
    const auto sequentialLatency = static_cast<double>(singleTuple->medianLatency.count()) * static_cast<double>(numberOfPayloads);
    const auto batchedLatency = static_cast<double>(batched->medianLatency.count());
    const auto speedup = batchedLatency > 0 ? sequentialLatency / batchedLatency : 0.0;
    const auto implementation = speedup >= configuration.postJoinInferenceMinimumSpeedup.getValue()
        ? PostJoinInferenceImplementation::BATCHED
        : PostJoinInferenceImplementation::SEQUENTIAL;
    NES_INFO(
        "Post-join inference of model '{}' on worker '{}': {} x batch-1 invocation = {:.0f} ns, one batch-{} invocation = {:.0f} ns, "
        "speedup {:.3f} (minimum {:.3f}) -> {}",
        modelName,
        worker,
        numberOfPayloads,
        sequentialLatency,
        numberOfPayloads,
        batchedLatency,
        speedup,
        configuration.postJoinInferenceMinimumSpeedup.getValue(),
        magic_enum::enum_name(implementation));

    /// The batched alternative invokes the model once per joined record with all its payloads, the sequential one with single tuples.
    assessMemory(worker, inferModel, implementation == PostJoinInferenceImplementation::BATCHED ? numberOfPayloads : 1);
    return implementation;
}

uint64_t
InferenceImplementationSelector::selectBatchSize(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const
{
    const auto fixedBatchSize = configuration.inferenceBatchSize.getValue();
    const auto& modelName = inferModel.getModel().getName();
    const auto window = findFeedingWindow(logicalOperator);
    if (!window.has_value())
    {
        NES_INFO("Inference of model '{}' is not fed by a time window, using the configured batch size {}", modelName, fixedBatchSize);
        return fixedBatchSize;
    }
    const auto windowType = std::dynamic_pointer_cast<Windowing::TimeBasedWindowType>(window->getWindowType());
    if (window->isKeyed() || !windowType)
    {
        NES_WARNING(
            "Inference of model '{}' is fed by a {} window, whose result rate is unknown; using the configured batch size {}",
            modelName,
            window->isKeyed() ? "keyed" : "non-time-based",
            fixedBatchSize);
        return fixedBatchSize;
    }

    const auto placement = getTrait<PlacementTrait>(logicalOperator.getTraitSet());
    if (!profiler || !placement.has_value())
    {
        NES_WARNING(
            "Cannot profile batched inference of model '{}' ({}), using the configured batch size {}",
            modelName,
            profiler ? "operator is not placed" : "no inference profiler is available",
            fixedBatchSize);
        return fixedBatchSize;
    }

    /// Window results arrive once per stride of event time; the replay speedup converts the stride and size to wall-clock time.
    const auto replaySpeedup = static_cast<double>(configuration.inferenceReplaySpeedup.getValue());
    PRECONDITION(replaySpeedup > 0, "The inference replay speedup must be positive, but is {}", replaySpeedup);
    const auto toWallClock = [replaySpeedup](const Windowing::TimeMeasure& measure)
    {
        return std::chrono::nanoseconds{static_cast<int64_t>(
            static_cast<double>(std::chrono::nanoseconds{std::chrono::milliseconds{measure.getTime()}}.count()) / replaySpeedup)};
    };
    const auto stride = toWallClock(windowType->getSlide());
    const auto size = toWallClock(windowType->getSize());
    PRECONDITION(stride.count() > 0, "Expected a window with a positive stride");

    /// The fill time (B - 1) * S alone must stay below the window size, which bounds the candidates before profiling.
    const auto largestFillableBatch = static_cast<size_t>((size.count() + stride.count() - 1) / stride.count());
    const auto largestCandidate
        = std::max<size_t>(1, std::min<size_t>(configuration.inferenceMaxBatchSize.getValue(), largestFillableBatch));
    std::vector<size_t> batchSizes;
    for (size_t batchSize = 1; batchSize <= largestCandidate; batchSize *= 2)
    {
        batchSizes.push_back(batchSize);
    }

    const auto& worker = placement.value()->onNode;
    const auto profile = getProfile(worker, inferModel, batchSizes);
    if (!profile.has_value())
    {
        NES_WARNING(
            "Profiling model '{}' on worker '{}' failed ({}), using the configured batch size {}",
            modelName,
            worker,
            profile.error().what(),
            fixedBatchSize);
        return fixedBatchSize;
    }

    std::vector<BatchSizeCandidate> candidates;
    for (const auto batchSize : batchSizes)
    {
        const auto batch = profile->forBatchSize(batchSize);
        INVARIANT(batch.has_value(), "Expected a profile for batch size {}", batchSize);
        const auto invocationLatency = batch->medianLatency;
        const auto resultLatency = stride * static_cast<int64_t>(batchSize - 1) + invocationLatency;
        candidates.push_back(BatchSizeCandidate{
            .batchSize = batchSize,
            .invocationLatency = invocationLatency,
            .resultLatency = resultLatency,
            .throughput = invocationLatency.count() > 0
                ? static_cast<double>(batchSize) * 1e9 / static_cast<double>(invocationLatency.count())
                : std::numeric_limits<double>::infinity(),
            .meetsHardSlo = invocationLatency <= stride,
            .meetsWeakSlo = resultLatency <= size});
        NES_INFO(
            "Batch size {} for model '{}' on worker '{}': invocation {} us (hard SLO {} us: {}), result latency {} us (weak SLO {} us: "
            "{}), throughput {:.0f} tuples/s",
            batchSize,
            modelName,
            worker,
            invocationLatency.count() / 1000,
            stride.count() / 1000,
            candidates.back().meetsHardSlo ? "met" : "violated",
            resultLatency.count() / 1000,
            size.count() / 1000,
            candidates.back().meetsWeakSlo ? "met" : "violated",
            candidates.back().throughput);
    }

    /// Prefer candidates meeting both SLOs, then the hard one only, then any; among them, the highest throughput wins and ties go to
    /// the smaller batch, which has the lower latency.
    const auto selectFrom = [&candidates](const auto& admissible) -> std::optional<BatchSizeCandidate>
    {
        std::optional<BatchSizeCandidate> selected;
        for (const auto& candidate : candidates | std::views::filter(admissible))
        {
            if (!selected.has_value() || candidate.throughput > selected->throughput)
            {
                selected = candidate;
            }
        }
        return selected;
    };
    const auto selectedBatchSize = [&]() -> uint64_t
    {
        if (const auto selected
            = selectFrom([](const BatchSizeCandidate& candidate) { return candidate.meetsHardSlo && candidate.meetsWeakSlo; }))
        {
            NES_INFO(
                "Selected batch size {} for model '{}' on worker '{}': meets both latency SLOs", selected->batchSize, modelName, worker);
            return selected->batchSize;
        }
        if (const auto selected = selectFrom([](const BatchSizeCandidate& candidate) { return candidate.meetsHardSlo; }))
        {
            NES_WARNING(
                "Selected batch size {} for model '{}' on worker '{}': no batch size meets the weak latency SLO",
                selected->batchSize,
                modelName,
                worker);
            return selected->batchSize;
        }
        const auto selected = selectFrom([](const BatchSizeCandidate&) { return true; });
        NES_WARNING(
            "Selected batch size {} for model '{}' on worker '{}': no batch size meets the hard latency SLO, choosing the highest "
            "throughput",
            selected->batchSize,
            modelName,
            worker);
        return selected->batchSize;
    }();

    assessMemory(worker, inferModel, selectedBatchSize);
    return selectedBatchSize;
}

std::optional<InferenceMemoryProfile>
InferenceImplementationSelector::getMemoryProfile(const Host& worker, const RegisteredModel& model) const
{
    const std::scoped_lock lock(profileCache->mutex);
    const auto cached = profileCache->memoryProfiles.find(std::make_tuple(worker.getRawValue(), model.getName(), model.getPath().string()));
    if (cached == profileCache->memoryProfiles.end())
    {
        return std::nullopt;
    }
    return cached->second;
}

std::expected<ModelProfile, Exception> InferenceImplementationSelector::getProfile(
    const Host& worker, const InferModelLogicalOperator& inferModel, const std::vector<size_t>& batchSizes) const
{
    const auto& model = inferModel.getModel();
    const auto key = [&](const size_t batchSize)
    { return std::make_tuple(worker.getRawValue(), model.getName(), model.getPath().string(), batchSize); };

    const std::scoped_lock lock(profileCache->mutex);
    const auto missingBatchSizes = batchSizes
        | std::views::filter([&](const size_t batchSize) { return !profileCache->profiles.contains(key(batchSize)); })
        | std::ranges::to<std::vector>();
    if (!missingBatchSizes.empty())
    {
        auto profile = profiler->profile(
            worker,
            ModelProfilingRequest{
                .model = model.getImported(),
                .batchSizes = missingBatchSizes,
                .warmupIterations = configuration.inferenceProfilingWarmupIterations.getValue(),
                .measurementIterations = configuration.inferenceProfilingIterations.getValue(),
                .assessMemoryForBatchSize = std::nullopt});
        if (!profile.has_value())
        {
            return std::unexpected(profile.error());
        }
        for (const auto& batch : profile->batches)
        {
            profileCache->profiles.insert_or_assign(key(batch.batchSize), batch);
        }
    }

    ModelProfile profile;
    for (const auto batchSize : batchSizes)
    {
        if (const auto cached = profileCache->profiles.find(key(batchSize)); cached != profileCache->profiles.end())
        {
            profile.batches.push_back(cached->second);
        }
    }
    return profile;
}

void InferenceImplementationSelector::assessMemory(
    const Host& worker, const InferModelLogicalOperator& inferModel, const size_t deployedBatchSize) const
{
    const auto& model = inferModel.getModel();
    const auto memoryKey = std::make_tuple(worker.getRawValue(), model.getName(), model.getPath().string());

    const std::scoped_lock lock(profileCache->mutex);
    if (profileCache->memoryProfiles.contains(memoryKey))
    {
        return;
    }

    auto profile = profiler->profile(
        worker,
        ModelProfilingRequest{
            .model = model.getImported(),
            .batchSizes = {},
            .warmupIterations = 0,
            .measurementIterations = 1,
            .assessMemoryForBatchSize = deployedBatchSize});
    if (!profile.has_value())
    {
        NES_WARNING("Assessing the memory of model '{}' on worker '{}' failed: {}", model.getName(), worker, profile.error().what());
        return;
    }
    profileCache->memoryProfiles.insert_or_assign(memoryKey, profile->memory);
    logMemoryProfile(worker, model.getName(), profile->memory);
}
}
