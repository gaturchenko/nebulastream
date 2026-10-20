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

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
#include <ModelCatalog.hpp>
#include <ModelProfile.hpp>
#include <QueryOptimizerConfiguration.hpp>

namespace NES
{

/// Selects the physical implementation of inference operators from model profiles of the worker each operator is placed on.
/// Runs after placement, so that the profiling job targets the worker that executes the operator, and only annotates traits,
/// leaving the plan structure untouched.
///
/// Post-join batching: an inference operator over k compatible payload fields of a joined record either invokes the model once
/// with a batch of k payloads or k times with a single payload. Under OPTIMIZER_CHOOSES, the worker profiles both invocations,
/// and batching is selected if k single-tuple invocations take at least the configured minimum speedup longer than one batch-k
/// invocation.
///
/// Batch size: an inference operator consuming the results of an unkeyed time window receives one result per window stride S.
/// Batching B results therefore delays the first of them by the fill time (B - 1) * S before the batch is invoked in t(B). Two
/// constraints bound the batch size from opposite sides. A batch of B results arrives every B * S, so a batch size keeps up only if
/// its utilization t(B) / (B * S) stays below the configured maximum, which leaves headroom for queueing; small batches that cannot
/// amortize the per-invocation cost fall behind at high rates. And a result must be inferred within the window size, the latency
/// SLO (B - 1) * S + t(B) <= W, which large batches miss at low rates. Under OPTIMIZER_CHOOSES, the worker profiles t(B) for
/// power-of-two candidates, and among those meeting both constraints the batch size with the highest throughput B / t(B) is
/// selected. If none does, the batch size with the lowest result latency among those that keep up is selected, and if none keeps
/// up, the one with the highest throughput.
///
/// The inference sequence operator, inserted before the batch size is known to restore the record order of batched inference, is
/// removed below inference operators that end up invoking the model with single tuples.
///
/// Once an operator's invocation is decided, the worker assesses what the model costs in memory there, using sessions set up for
/// the decided batch size. Under OPTIMIZER_CHOOSES, that assessment then decides the operator's deployment: whether the worker
/// threads share one compiled model, and whether they share one prediction cache. Both alternatives trade speed for memory, so
/// the fastest combination that fits the worker's memory budget is selected.
class InferenceImplementationSelector
{
public:
    InferenceImplementationSelector(QueryOptimizerConfiguration configuration, std::shared_ptr<InferenceProfiler> profiler);

    [[nodiscard]] LogicalPlan apply(const LogicalPlan& plan) const;

    /// What inference with `model` costs in memory on `worker`, once a profiling job has assessed it.
    [[nodiscard]] std::optional<InferenceMemoryProfile> getMemoryProfile(const Host& worker, const RegisteredModel& model) const;

private:
    /// Profiles are reused across queries that place the same model on the same worker: invocation latencies per batch size, and
    /// the memory assessment, which the worker only performs while it is not cached yet.
    struct ProfileCache
    {
        std::mutex mutex;
        std::map<std::tuple<std::string, std::string, std::string, size_t>, BatchProfile> profiles;
        std::map<std::tuple<std::string, std::string, std::string>, InferenceMemoryProfile> memoryProfiles;
    };

    [[nodiscard]] LogicalOperator apply(const LogicalOperator& logicalOperator) const;
    [[nodiscard]] PostJoinInferenceImplementation
    selectPostJoinImplementation(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const;
    [[nodiscard]] uint64_t selectBatchSize(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const;
    /// Assesses what the model costs in memory on the worker, with runtime sessions set up for the batch size the operator was
    /// decided to use. Runs once per model and worker.
    void assessMemory(const Host& worker, const InferModelLogicalOperator& inferModel, size_t deployedBatchSize) const;
    /// Chooses whether the worker threads share one compiled model and one prediction cache, from what the alternatives cost on
    /// the worker and what memory it has.
    void selectDeployment(
        const LogicalOperator& logicalOperator,
        const InferModelLogicalOperator& inferModel,
        size_t deployedBatchSize,
        TraitSet& traitSet) const;
    [[nodiscard]] std::expected<ModelProfile, Exception>
    getProfile(const Host& worker, const InferModelLogicalOperator& inferModel, const std::vector<size_t>& batchSizes) const;

    QueryOptimizerConfiguration configuration;
    std::shared_ptr<InferenceProfiler> profiler;
    std::shared_ptr<ProfileCache> profileCache;
};

}
