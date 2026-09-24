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

#include <ModelProfileSerialization.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <vector>

#include <Util/Reflection.hpp>
#include <rfl/json/read.hpp>
#include <rfl/json/write.hpp>
#include <ErrorHandling.hpp>
#include <Model.hpp>
#include <ModelProfile.hpp>
#include <SingleNodeWorkerRPCService.pb.h>

namespace NES
{

void serializeModelProfilingRequest(const ModelProfilingRequest& request, ProfileModelRequest* serialized)
{
    serialized->set_reflectedmodel(rfl::json::write(reflect(request.model)));
    for (const auto batchSize : request.batchSizes)
    {
        serialized->add_batchsizes(batchSize);
    }
    serialized->set_warmupiterations(request.warmupIterations);
    serialized->set_measurementiterations(request.measurementIterations);
    if (request.assessMemoryForBatchSize.has_value())
    {
        serialized->set_assessmemoryforbatchsize(request.assessMemoryForBatchSize.value());
    }
}

ModelProfilingRequest deserializeModelProfilingRequest(const ProfileModelRequest& serialized)
{
    auto reflectedModel = rfl::json::read<Reflected>(serialized.reflectedmodel());
    if (!reflectedModel.has_value())
    {
        throw CannotDeserialize("Failed to read the model of a profiling request: {}", reflectedModel.error().what());
    }
    return ModelProfilingRequest{
        .model = unreflect<ImportedModel>(reflectedModel.value()),
        .batchSizes = serialized.batchsizes() | std::views::transform([](const uint64_t size) { return static_cast<size_t>(size); })
            | std::ranges::to<std::vector>(),
        .warmupIterations = serialized.warmupiterations(),
        .measurementIterations = serialized.measurementiterations(),
        .assessMemoryForBatchSize = serialized.has_assessmemoryforbatchsize()
            ? std::optional{static_cast<size_t>(serialized.assessmemoryforbatchsize())}
            : std::nullopt};
}

void serializeModelProfile(const ModelProfile& profile, ProfileModelReply* serialized)
{
    for (const auto& batch : profile.batches)
    {
        auto* batchProfile = serialized->add_batches();
        batchProfile->set_batchsize(batch.batchSize);
        batchProfile->set_iterations(batch.iterations);
        batchProfile->set_medianlatencyinnanoseconds(batch.medianLatency.count());
        batchProfile->set_meanlatencyinnanoseconds(batch.meanLatency.count());
        batchProfile->set_minlatencyinnanoseconds(batch.minLatency.count());
    }

    auto* sessionMemory = serialized->mutable_sessionmemory();
    sessionMemory->set_measuredbatchsize(profile.memory.session.measuredBatchSize);
    sessionMemory->set_modelpayloadbytes(profile.memory.session.modelPayloadBytes);
    sessionMemory->set_inputbufferbytes(profile.memory.session.inputBufferBytes);
    sessionMemory->set_outputbufferbytes(profile.memory.session.outputBufferBytes);
    sessionMemory->set_firstprivatesessionbytes(profile.memory.session.firstPrivateSessionBytes);
    sessionMemory->set_additionalprivatesessionbytes(profile.memory.session.additionalPrivateSessionBytes);
    sessionMemory->set_firstsharedsessionbytes(profile.memory.session.firstSharedSessionBytes);
    sessionMemory->set_additionalsharedsessionbytes(profile.memory.session.additionalSharedSessionBytes);
    sessionMemory->set_measured(profile.memory.session.measured);

    auto* cacheMemory = serialized->mutable_predictioncachememory();
    cacheMemory->set_entries(profile.memory.predictionCache.entries);
    cacheMemory->set_bytesperentry(profile.memory.predictionCache.bytesPerEntry);
    cacheMemory->set_threadlocalbytes(profile.memory.predictionCache.threadLocalBytes);
    cacheMemory->set_globalbytes(profile.memory.predictionCache.globalBytes);

    auto* workerMemory = serialized->mutable_workermemory();
    workerMemory->set_physicalmemorybytes(profile.memory.worker.physicalMemoryBytes);
    workerMemory->set_bufferpoolbytes(profile.memory.worker.bufferPoolBytes);
    workerMemory->set_numberofworkerthreads(profile.memory.worker.numberOfWorkerThreads);
}

ModelProfile deserializeModelProfile(const ProfileModelReply& serialized)
{
    ModelProfile profile;
    for (const auto& batch : serialized.batches())
    {
        profile.batches.push_back(BatchProfile{
            .batchSize = batch.batchsize(),
            .iterations = batch.iterations(),
            .medianLatency = std::chrono::nanoseconds{batch.medianlatencyinnanoseconds()},
            .meanLatency = std::chrono::nanoseconds{batch.meanlatencyinnanoseconds()},
            .minLatency = std::chrono::nanoseconds{batch.minlatencyinnanoseconds()}});
    }

    profile.memory.session = SessionMemoryProfile{
        .measuredBatchSize = serialized.sessionmemory().measuredbatchsize(),
        .modelPayloadBytes = serialized.sessionmemory().modelpayloadbytes(),
        .inputBufferBytes = serialized.sessionmemory().inputbufferbytes(),
        .outputBufferBytes = serialized.sessionmemory().outputbufferbytes(),
        .firstPrivateSessionBytes = serialized.sessionmemory().firstprivatesessionbytes(),
        .additionalPrivateSessionBytes = serialized.sessionmemory().additionalprivatesessionbytes(),
        .firstSharedSessionBytes = serialized.sessionmemory().firstsharedsessionbytes(),
        .additionalSharedSessionBytes = serialized.sessionmemory().additionalsharedsessionbytes(),
        .measured = serialized.sessionmemory().measured()};

    profile.memory.predictionCache = PredictionCacheMemoryProfile{
        .entries = serialized.predictioncachememory().entries(),
        .bytesPerEntry = serialized.predictioncachememory().bytesperentry(),
        .threadLocalBytes = serialized.predictioncachememory().threadlocalbytes(),
        .globalBytes = serialized.predictioncachememory().globalbytes()};

    profile.memory.worker = WorkerMemoryProfile{
        .physicalMemoryBytes = serialized.workermemory().physicalmemorybytes(),
        .bufferPoolBytes = serialized.workermemory().bufferpoolbytes(),
        .numberOfWorkerThreads = serialized.workermemory().numberofworkerthreads()};

    return profile;
}

}
