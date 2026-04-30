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

#include <DataTypes/DataTypeProvider.hpp>
#include <Functions/FunctionProvider.hpp>
#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <RewriteRules/AbstractRewriteRule.hpp>
#include <Traits/OutputOriginIdsTrait.hpp>
#include <BatchingPhysicalOperator.hpp>
#include <BatchCacheInferenceOperator.hpp>
#include <BatchInferenceOperator.hpp>
#include <BatchInferenceOperatorHandler.hpp>
#include <CacheInferenceOperator.hpp>
#include <InferenceOperator.hpp>
#include <InferenceOperatorHandler.hpp>
#include <InferModelLogicalOperator.hpp>
#include <InterBufferBatchingPhysicalOperator.hpp>
#include <Nautilus/Interface/Hash/MurMur3HashFunction.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedEntryMemoryProvider.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <HashMapOptions.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <RewriteRuleRegistry.hpp>

struct LowerToPhysicalInferenceOperator : NES::AbstractRewriteRule
{
    bool findAggregationRecursively(const NES::LogicalOperator& child)
    {
        const auto children = child.getChildren();

        if (children.empty())
        {
            return false;
        }

        const auto& firstChild = children.at(0);

        if (firstChild.tryGetAs<NES::WindowedAggregationLogicalOperator>())
        {
            return true;
        }

        return findAggregationRecursively(firstChild);
    }

    NES::HashMapOptions createHashMapOptions(
        std::vector<NES::PhysicalFunction> keyFunctions,
        NES::Schema& inputSchema,
        const NES::QueryExecutionConfiguration& conf)
    {
        std::vector<std::string> fieldKeyNames = inputSchema.getFieldNames();
        const uint64_t valueSize = NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT64).getSizeInBytes() * 2;
        const uint64_t keySize = inputSchema.getSizeOfSchemaInBytes();

        const auto pageSize = conf.pageSize.getValue();
        const auto numberOfBuckets = conf.numberOfPartitions.getValue();
        const auto entrySize = sizeof(NES::ChainedHashMapEntry) + keySize + valueSize;
        const auto entriesPerPage = pageSize / entrySize;

        const auto& [fieldKeys, fieldValues] =
            NES::ChainedEntryMemoryProvider::createFieldOffsets(inputSchema.addField("rowInputIndex", NES::DataType::Type::INT64)
                .addField("rowOutputIndex", NES::DataType::Type::INT64), fieldKeyNames, {"rowInputIndex", "rowOutputIndex"});

        NES::HashMapOptions hashMapOptions{
            std::make_unique<NES::MurMur3HashFunction>(),
            std::move(keyFunctions),
            fieldKeys,
            fieldValues,
            entriesPerPage,
            entrySize,
            keySize,
            valueSize,
            pageSize,
            numberOfBuckets};
        return hashMapOptions;
    }

    explicit LowerToPhysicalInferenceOperator(NES::QueryExecutionConfiguration conf) : conf(std::move(conf)) { }
    NES::RewriteRuleResultSubgraph apply(NES::LogicalOperator logicalOperator) override
    {
        auto inferModelOperator = logicalOperator.getAs<NES::InferModel::InferModelLogicalOperator>();

        const auto& model = inferModelOperator->getModel();
        auto handlerId = NES::getNextOperatorHandlerId();

        auto inputSchema = logicalOperator->getInputSchemas().at(0);

        auto inputFunctions = std::views::transform(
                                  inferModelOperator->getInputFields(),
                                  [](const auto& function) { return NES::QueryCompilation::FunctionProvider::lowerFunction(function); })
            | std::ranges::to<std::vector>();
        auto outputNames = model.getOutputs() | std::views::keys | std::ranges::to<std::vector>();

        const auto batchSize = conf.inferenceConfiguration.batchSize;
        const auto useBatchDeduplication = conf.inferenceConfiguration.useBatchDeduplication;
        const auto predictionCacheType = conf.inferenceConfiguration.predictionCacheType;
        const auto predictionCacheSize = conf.inferenceConfiguration.numberOfEntriesPredictionCache;
        const auto openVinoInferenceNumThreads = conf.inferenceConfiguration.openvinoInferenceNumThreads.getValue();
        const auto openVinoNumStreams = conf.inferenceConfiguration.openvinoNumStreams.getValue();
        const auto openVinoEnableCpuPinning = conf.inferenceConfiguration.openvinoEnableCpuPinning.getValue();

        const auto inputDtype = model.getInputDtype();
        const auto outputDtype = model.getOutputDtype();
        const auto isVarSizedInput = inferModelOperator->getInputFields().size() == 1
            && inferModelOperator->getInputFields().at(0).getDataType().type == NES::DataType::Type::VARSIZED;
        const auto isVarSizedOutput = model.getOutputs().size() == 1 && model.getOutputs().at(0).second.type == NES::DataType::Type::VARSIZED;

        /// if the batch size is 1, then we simply use the inference operator with PipelineLocation::INTERMEDIATE
        /// else, add the batching operator (custom emit) and batch inference operator (custom scan)
        std::shared_ptr<NES::PhysicalOperatorWrapper> wrapper = nullptr;
        auto runtimeConfiguration = NES::InferenceRuntimeConfiguration{
            .openVinoInferenceNumThreads = openVinoInferenceNumThreads,
            .openVinoNumStreams = openVinoNumStreams,
            .openVinoEnableCpuPinning = openVinoEnableCpuPinning};

        if (batchSize.getValue() == 1)
        {
            auto handler = std::make_shared<NES::InferenceOperatorHandler>(model, runtimeConfiguration);

            switch (predictionCacheType.getValue())
            {
                case NES::Configurations::PredictionCacheType::NONE: {
                    NES_DEBUG("Lower InferModel operator to InferenceOperator");
                    auto inferenceOperator = NES::InferenceOperator(handlerId, inputFunctions, outputNames, inputDtype, outputDtype);
                    inferenceOperator.isVarSizedInput = isVarSizedInput;
                    inferenceOperator.isVarSizedOutput = isVarSizedOutput;
                    inferenceOperator.outputSize = model.outputSize();
                    inferenceOperator.inputSize = model.inputSize();

                    wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                        inferenceOperator,
                        logicalOperator->getInputSchemas().at(0),
                        logicalOperator->getOutputSchema(),
                        handlerId,
                        std::move(handler),
                        NES::PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);
                    return {wrapper, {wrapper}};
                }

                case NES::Configurations::PredictionCacheType::TWO_QUEUES:
                case NES::Configurations::PredictionCacheType::FIFO:
                case NES::Configurations::PredictionCacheType::LFU:
                case NES::Configurations::PredictionCacheType::LRU:
                case NES::Configurations::PredictionCacheType::SECOND_CHANCE:
                case NES::Configurations::PredictionCacheType::ALWAYS_MISS: {
                    NES_DEBUG("Lower InferModel operator to CacheInferenceOperator");
                    NES::Configurations::PredictionCacheOptions predictionCacheOptions{
                        predictionCacheType.getValue(),
                        predictionCacheSize.getValue()};
                    auto cacheOperator = NES::CacheInferenceOperator(handlerId, inputFunctions, outputNames, predictionCacheOptions, inputDtype, outputDtype);
                    cacheOperator.isVarSizedInput = isVarSizedInput;
                    cacheOperator.isVarSizedOutput = isVarSizedOutput;
                    cacheOperator.outputSize = model.outputSize();
                    cacheOperator.inputSize = model.inputSize();

                    wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                        cacheOperator,
                        logicalOperator->getInputSchemas().at(0),
                        logicalOperator->getOutputSchema(),
                        handlerId,
                        std::move(handler),
                        NES::PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);
                    return {wrapper, {wrapper}};
                }
            }
        }
        else
        {
            auto outputOriginIdsOpt = getTrait<NES::OutputOriginIdsTrait>(logicalOperator->getTraitSet());
            auto inputOriginIdsOpt = getTrait<NES::OutputOriginIdsTrait>(logicalOperator->getChildren().at(0).getTraitSet());
            PRECONDITION(outputOriginIdsOpt.has_value(), "Expected the outputOriginIds trait to be set");
            PRECONDITION(inputOriginIdsOpt.has_value(), "Expected the inputOriginIds trait to be set");

            auto& outputOriginIds = outputOriginIdsOpt.value();
            auto outputOriginId = outputOriginIds[0];
            auto inputOriginIds = inputOriginIdsOpt.value();

            const auto pageSize = conf.pageSize.getValue();

            auto memoryProvider = NES::TupleBufferRef::create(pageSize, inputSchema);
            auto handler = std::make_shared<NES::BatchInferenceOperatorHandler>(
                inputOriginIds | std::ranges::to<std::vector>(), outputOriginId, model, batchSize.getValue(), runtimeConfiguration);

            std::shared_ptr<NES::PhysicalOperatorWrapper> batchingWrapper = nullptr;

            const auto child = inferModelOperator.getChildren().at(0);
            if (findAggregationRecursively(child))
            {
                auto batchingOperator = NES::InterBufferBatchingPhysicalOperator(handlerId, memoryProvider);
                batchingWrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                    batchingOperator,
                    inputSchema,
                    inputSchema,
                    handlerId,
                    handler,
                    NES::PhysicalOperatorWrapper::PipelineLocation::EMIT);
            }
            else
            {
                auto batchingOperator = NES::BatchingPhysicalOperator(handlerId, memoryProvider);
                batchingWrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                    batchingOperator,
                    inputSchema,
                    inputSchema,
                    handlerId,
                    handler,
                    NES::PhysicalOperatorWrapper::PipelineLocation::EMIT);
            }

            switch (predictionCacheType.getValue())
            {
                case NES::Configurations::PredictionCacheType::NONE: {
                    NES_DEBUG("Lower InferModel operator to BatchInferenceOperator");

                    auto hashMapOptions = createHashMapOptions(inputFunctions, inputSchema, conf);
                    auto batchOperator = NES::BatchInferenceOperator(
                        handlerId,
                        inputFunctions,
                        outputNames,
                        memoryProvider,
                        inputDtype,
                        outputDtype,
                        hashMapOptions,
                        useBatchDeduplication.getValue());
                    batchOperator.isVarSizedInput = isVarSizedInput;
                    batchOperator.isVarSizedOutput = isVarSizedOutput;
                    batchOperator.outputSize = model.outputSize();
                    batchOperator.inputSize = model.inputSize();

                    wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                        batchOperator,
                        inputSchema,
                        logicalOperator->getOutputSchema(),
                        handlerId,
                        handler,
                        NES::PhysicalOperatorWrapper::PipelineLocation::SCAN,
                        std::vector{batchingWrapper});
                    return {wrapper, {batchingWrapper}};
                }
                case NES::Configurations::PredictionCacheType::TWO_QUEUES:
                case NES::Configurations::PredictionCacheType::FIFO:
                case NES::Configurations::PredictionCacheType::LFU:
                case NES::Configurations::PredictionCacheType::LRU:
                case NES::Configurations::PredictionCacheType::SECOND_CHANCE:
                case NES::Configurations::PredictionCacheType::ALWAYS_MISS: {
                    NES_DEBUG("Lower InferModel operator to BatchCacheInferenceOperator");
                    NES::Configurations::PredictionCacheOptions predictionCacheOptions{
                        predictionCacheType.getValue(),
                        predictionCacheSize.getValue()};
                    auto hashMapOptions = createHashMapOptions(inputFunctions, inputSchema, conf);
                    auto batchCacheOperator = NES::BatchCacheInferenceOperator(
                        handlerId,
                        inputFunctions,
                        outputNames,
                        memoryProvider,
                        predictionCacheOptions,
                        inputDtype,
                        outputDtype,
                        hashMapOptions,
                        useBatchDeduplication.getValue());
                    batchCacheOperator.isVarSizedInput = isVarSizedInput;
                    batchCacheOperator.isVarSizedOutput = isVarSizedOutput;
                    batchCacheOperator.outputSize = model.outputSize();
                    batchCacheOperator.inputSize = model.inputSize();

                    wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                        batchCacheOperator,
                        inputSchema,
                        logicalOperator->getOutputSchema(),
                        handlerId,
                        handler,
                        NES::PhysicalOperatorWrapper::PipelineLocation::SCAN,
                        std::vector{batchingWrapper});
                    return {wrapper, {batchingWrapper}};
                }
            }
        }
    }
private:
    NES::QueryExecutionConfiguration conf;
};

std::unique_ptr<NES::AbstractRewriteRule>
NES::RewriteRuleGeneratedRegistrar::RegisterInferenceModelRewriteRule(RewriteRuleRegistryArguments arguments)
{
    return std::make_unique<LowerToPhysicalInferenceOperator>(arguments.conf);
}
