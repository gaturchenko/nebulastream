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

#include <QueryManager/EmbeddedWorkerQuerySubmissionBackend.hpp>

#include <chrono>
#include <memory>
#include <utility>
#include <Identifiers/Identifiers.hpp>
#include <Listeners/QueryLog.hpp>
#include <Plans/LogicalPlan.hpp>
#include <QueryManager/QueryManager.hpp>

#include <Util/Logger/impl/NesLogger.hpp>
#include <cpptrace/from_current.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiling.hpp>
#include <ModelProfile.hpp>
#include <QueryStatus.hpp>
#include <SingleNodeWorkerConfiguration.hpp>
#include <WorkerStatus.hpp>

namespace NES
{

namespace
{
/// Start with the per-worker topology config, then overlay only
/// explicitly-set CLI values so that CLI args take highest priority
/// but topology values aren't clobbered by CLI defaults.
SingleNodeWorkerConfiguration mergeWorkerConfiguration(const WorkerConfig& config, const SingleNodeWorkerConfiguration& workerConfiguration)
{
    SingleNodeWorkerConfiguration mergedConfig = config.config;
    mergedConfig.applyExplicitlySetFrom(workerConfiguration);
    return mergedConfig;
}
}

EmbeddedWorkerQuerySubmissionBackend::EmbeddedWorkerQuerySubmissionBackend(
    WorkerConfig config, SingleNodeWorkerConfiguration workerConfiguration)
    : worker{[&]()
             {
                 SingleNodeWorkerConfiguration mergedConfig = mergeWorkerConfiguration(config, workerConfiguration);

                 /// Set grpc/data from topology (these always come from cluster config)
                 mergedConfig.grpcAddressUri.setValue(config.host.getRawValue());
                 mergedConfig.dataAddress.setValue(config.dataAddress);

                 const LogContext logContext("create", config.host);
                 return SingleNodeWorker(mergedConfig, config.host);
             }()}
{
}

std::expected<QueryId, Exception> EmbeddedWorkerQuerySubmissionBackend::registerQuery(LogicalPlan plan)
{
    return worker.registerQuery(plan);
}

std::expected<void, Exception> EmbeddedWorkerQuerySubmissionBackend::start(QueryId queryId)
{
    return worker.startQuery(queryId);
}

std::expected<void, Exception> EmbeddedWorkerQuerySubmissionBackend::stop(QueryId queryId)
{
    return worker.stopQuery(queryId);
}

std::expected<LocalQueryStatusSnapshot, Exception> EmbeddedWorkerQuerySubmissionBackend::status(QueryId queryId) const
{
    return worker.getQueryStatus(queryId);
}

std::expected<WorkerStatus, Exception> EmbeddedWorkerQuerySubmissionBackend::workerStatus(std::chrono::system_clock::time_point after) const
{
    return worker.getWorkerStatus(after);
}

std::expected<ModelProfile, Exception> EmbeddedWorkerQuerySubmissionBackend::profileModel(const ModelProfilingRequest& request) const
{
    return worker.profileModel(request);
}

BackendProvider createEmbeddedBackend(const SingleNodeWorkerConfiguration& workerConfiguration)
{
    return [workerConfiguration](const WorkerConfig& config) /// NOLINT(bugprone-exception-escape)
    { return std::make_unique<EmbeddedWorkerQuerySubmissionBackend>(config, workerConfiguration); };
}

EmbeddedInferenceProfiler::EmbeddedInferenceProfiler(
    SharedPtr<const WorkerCatalog> workerCatalog, SingleNodeWorkerConfiguration workerConfiguration)
    : workerCatalog(std::move(workerCatalog)), workerConfiguration(std::move(workerConfiguration))
{
}

std::expected<ModelProfile, Exception> EmbeddedInferenceProfiler::profile(const Host& worker, const ModelProfilingRequest& request)
{
    const auto config = workerCatalog->getWorker(worker);
    if (!config.has_value())
    {
        return std::unexpected(UnknownWorker("Cannot profile a model on unknown worker '{}'", worker));
    }
    CPPTRACE_TRY
    {
        const auto mergedConfig = mergeWorkerConfiguration(config.value(), workerConfiguration);
        return profileInferenceModel(
            request,
            mergedConfig.workerConfiguration.defaultQueryExecution,
            mergedConfig.workerConfiguration.queryEngine.numberOfWorkerThreads.getValue());
    }
    CPPTRACE_CATCH(...)
    {
        return std::unexpected(wrapExternalException());
    }
    std::unreachable();
}

}
