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

#include <cstddef>
#include <expected>
#include <memory>
#include <mutex>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/PlacementTrait.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/Logger.hpp>
#include <magic_enum/magic_enum.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
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
    if (const auto inferModel = logicalOperator.tryGetAs<InferModelLogicalOperator>();
        inferModel.has_value() && isPostJoinInference(inferModel.value().get()))
    {
        tryInsert(traitSet, PostJoinInferenceImplementationTrait{selectPostJoinImplementation(logicalOperator, inferModel.value().get())});
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
    const auto profile = getProfile(worker, inferModel, numberOfPayloads);
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
    return implementation;
}

std::expected<ModelProfile, Exception>
InferenceImplementationSelector::getProfile(const Host& worker, const InferModelLogicalOperator& inferModel, const size_t batchSize) const
{
    const auto& model = inferModel.getModel();
    auto key = std::make_tuple(worker.getRawValue(), model.getName(), model.getPath().string(), batchSize);
    const std::scoped_lock lock(profileCache->mutex);
    if (const auto cached = profileCache->profiles.find(key); cached != profileCache->profiles.end())
    {
        return cached->second;
    }

    auto profile = profiler->profile(
        worker,
        ModelProfilingRequest{
            .model = model.getImported(),
            .batchSizes = {1, batchSize},
            .warmupIterations = configuration.inferenceProfilingWarmupIterations.getValue(),
            .measurementIterations = configuration.inferenceProfilingIterations.getValue()});
    if (profile.has_value())
    {
        profileCache->profiles.emplace(std::move(key), profile.value());
    }
    return profile;
}

}
