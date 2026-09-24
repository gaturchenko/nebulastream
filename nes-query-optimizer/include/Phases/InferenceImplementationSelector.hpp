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
#include <string>
#include <tuple>
#include <vector>

#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
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
/// Batching B results therefore delays the first of them by the fill time (B - 1) * S before the batch is invoked in t(B). The
/// window defines two latency SLOs: a batched invocation must finish within the stride, t(B) <= S (hard, otherwise invocations
/// fall behind the arriving results), and a result must be inferred within the window size, (B - 1) * S + t(B) <= W (weak).
/// Under OPTIMIZER_CHOOSES, the worker profiles t(B) for power-of-two candidates, and the batch size with the highest throughput
/// B / t(B) that meets both SLOs is selected, relaxing the weak and then the hard SLO if no candidate meets them.
///
/// The inference sequence operator, inserted before the batch size is known to restore the record order of batched inference, is
/// removed below inference operators that end up invoking the model with single tuples.
class InferenceImplementationSelector
{
public:
    InferenceImplementationSelector(QueryOptimizerConfiguration configuration, std::shared_ptr<InferenceProfiler> profiler);

    [[nodiscard]] LogicalPlan apply(const LogicalPlan& plan) const;

private:
    /// Profiles are reused across queries that place the same model on the same worker, per batch size.
    struct ProfileCache
    {
        std::mutex mutex;
        std::map<std::tuple<std::string, std::string, std::string, size_t>, BatchProfile> profiles;
    };

    [[nodiscard]] LogicalOperator apply(const LogicalOperator& logicalOperator) const;
    [[nodiscard]] PostJoinInferenceImplementation
    selectPostJoinImplementation(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const;
    [[nodiscard]] uint64_t selectBatchSize(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModel) const;
    [[nodiscard]] std::expected<ModelProfile, Exception>
    getProfile(const Host& worker, const InferModelLogicalOperator& inferModel, const std::vector<size_t>& batchSizes) const;

    QueryOptimizerConfiguration configuration;
    std::shared_ptr<InferenceProfiler> profiler;
    std::shared_ptr<ProfileCache> profileCache;
};

}
