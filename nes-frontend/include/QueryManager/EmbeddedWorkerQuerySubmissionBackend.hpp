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

#include <QueryManager/QueryManager.hpp>

#include <chrono>
#include <Identifiers/Identifiers.hpp>
#include <Listeners/QueryLog.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Util/Pointers.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
#include <ModelProfile.hpp>
#include <QueryStatus.hpp>
#include <SingleNodeWorker.hpp>
#include <SingleNodeWorkerConfiguration.hpp>
#include <WorkerCatalog.hpp>
#include <WorkerStatus.hpp>

namespace NES
{
class EmbeddedWorkerQuerySubmissionBackend final : public QuerySubmissionBackend
{
public:
    EmbeddedWorkerQuerySubmissionBackend(WorkerConfig config, SingleNodeWorkerConfiguration workerConfiguration);
    [[nodiscard]] std::expected<QueryId, Exception> registerQuery(LogicalPlan) override;
    std::expected<void, Exception> start(QueryId) override;
    std::expected<void, Exception> stop(QueryId) override;
    [[nodiscard]] std::expected<LocalQueryStatusSnapshot, Exception> status(QueryId) const override;
    [[nodiscard]] std::expected<WorkerStatus, Exception> workerStatus(std::chrono::system_clock::time_point after) const override;
    [[nodiscard]] std::expected<ModelProfile, Exception> profileModel(const ModelProfilingRequest& request) const override;

private:
    SingleNodeWorker worker;
};

BackendProvider createEmbeddedBackend(const SingleNodeWorkerConfiguration& workerConfiguration);

/// Executes the optimizer's profiling jobs in-process with the configuration the embedded worker for the target host would be
/// created with. Allows profiling before any embedded worker exists, e.g., when queries are optimized ahead of their execution.
class EmbeddedInferenceProfiler final : public InferenceProfiler
{
    SharedPtr<const WorkerCatalog> workerCatalog;
    SingleNodeWorkerConfiguration workerConfiguration;

public:
    EmbeddedInferenceProfiler(SharedPtr<const WorkerCatalog> workerCatalog, SingleNodeWorkerConfiguration workerConfiguration);
    [[nodiscard]] std::expected<ModelProfile, Exception> profile(const Host& worker, const ModelProfilingRequest& request) override;
};

}
