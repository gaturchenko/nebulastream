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

#include <InferenceAdapter.hpp>
#include <InferenceOperatorHandler.hpp>
#include <PipelineExecutionContext.hpp>
#include <PredictionCache/PredictionCache.hpp>
#include <Util/Logger/Logger.hpp>
#include <algorithm>

namespace NES
{
namespace
{
constexpr uint64_t MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE = 4096;

uint64_t getPredictionCacheLookupIndexPageSize(const uint64_t keySize)
{
    const auto mapEntrySize = sizeof(ChainedHashMapEntry) + keySize + 2 * sizeof(uint64_t);
    return std::max(MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE, mapEntrySize);
}
}

InferenceOperatorHandler::InferenceOperatorHandler(Nebuli::Inference::Model model) : model(std::move(model))
{
}

void InferenceOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    threadLocalAdapters.reserve(pipelineExecutionContext.getNumberOfWorkerThreads());
    for (size_t threadId = 0; threadId < pipelineExecutionContext.getNumberOfWorkerThreads(); ++threadId)
    {
        threadLocalAdapters.emplace_back(InferenceAdapter::create());
        threadLocalAdapters.back()->initializeModel(model, 1);
    }
}
void InferenceOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext& pipelineExecutionContext)
{
    if (model.getInputs()[0].isType(DataType::Type::VARSIZED))
    {
        uint64_t misses{0};
        for (const auto& adapter : threadLocalAdapters) { misses += adapter->misses; }
        NES_INFO("{{\"pipeline_id\": {}, \"misses\": {}}}", pipelineExecutionContext.getPipelineId(), misses)
    }
    threadLocalAdapters.clear();
}

const Nebuli::Inference::Model& InferenceOperatorHandler::getModel() const
{
    return model;
}

const std::shared_ptr<InferenceAdapter>& InferenceOperatorHandler::getAdapter(WorkerThreadId threadId) const
{
    return threadLocalAdapters[threadId % threadLocalAdapters.size()];
}

void InferenceOperatorHandler::allocatePredictionCacheEntries(
    const uint64_t sizeOfEntry, const uint64_t numberOfEntries, AbstractBufferProvider* bufferProvider)
{
    if (hasPredictionCacheCreated.exchange(true))
    {
        return;
    }

    PRECONDITION(bufferProvider != nullptr, "Buffer provider should not be null");
    for (uint64_t i = 0; i < threadLocalAdapters.size(); ++i)
    {
        const auto neededSize = numberOfEntries * sizeOfEntry + sizeof(HitsAndMisses);
        INVARIANT(neededSize > 0, "Size of entry should be larger than 0");

        auto bufferOpt = bufferProvider->getUnpooledBuffer(neededSize);
        INVARIANT(bufferOpt.has_value(), "Buffer provider should return a buffer");
        std::ranges::fill(bufferOpt.value().getAvailableMemoryArea(), std::byte{0});
        predictionCacheEntriesBufferForWorkerThreads.emplace_back(bufferOpt.value());
        predictionCacheReplacementPosForWorkerThreads.emplace_back(uint64_t{0});

        const auto keySize = threadLocalAdapters.at(i)->inputSize;
        const auto pageSize = getPredictionCacheLookupIndexPageSize(keySize);
        predictionCacheLookupHashMapsForWorkerThreads.emplace_back(
            std::make_unique<ChainedHashMap>(keySize, 2 * sizeof(uint64_t), numberOfEntries, pageSize));
    }
}

const int8_t* InferenceOperatorHandler::getStartOfPredictionCacheEntries(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const
{
    PRECONDITION(threadLocalAdapters.size() > 0, "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntries = dynamic_cast<const StartPredictionCacheEntriesInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntries.workerThreadId % predictionCacheEntriesBufferForWorkerThreads.size();
    INVARIANT(
        not predictionCacheEntriesBufferForWorkerThreads.empty() and pos < predictionCacheEntriesBufferForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheEntriesBufferForWorkerThreads");

    return reinterpret_cast<const int8_t*>(predictionCacheEntriesBufferForWorkerThreads.at(pos).getAvailableMemoryArea().data());
}

uint64_t InferenceOperatorHandler::getReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const
{
    PRECONDITION(!threadLocalAdapters.empty(), "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntries = dynamic_cast<const StartPredictionCacheEntriesInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntries.workerThreadId % predictionCacheReplacementPosForWorkerThreads.size();
    INVARIANT(
        not predictionCacheReplacementPosForWorkerThreads.empty() and pos < predictionCacheReplacementPosForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheReplacementPosForWorkerThreads");
    return predictionCacheReplacementPosForWorkerThreads.at(pos);
}

void
InferenceOperatorHandler::setReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs, uint64_t idx)
{
    PRECONDITION(!threadLocalAdapters.empty(), "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntries = dynamic_cast<const StartPredictionCacheEntriesInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntries.workerThreadId % predictionCacheReplacementPosForWorkerThreads.size();
    INVARIANT(
        not predictionCacheReplacementPosForWorkerThreads.empty() and pos < predictionCacheReplacementPosForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheReplacementPosForWorkerThreads");
    predictionCacheReplacementPosForWorkerThreads[pos] = idx;
}

}
