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

#include <LoweringRules/LowerToPhysical/LowerToPhysicalInferModel.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include <DataTypes/DataTypeProvider.hpp>
#include <Inference/BatchCacheInferModelPhysicalOperator.hpp>
#include <Inference/BatchInferModelPhysicalOperator.hpp>
#include <Inference/BatchInferenceOperatorHandler.hpp>
#include <Inference/BatchingPhysicalOperator.hpp>
#include <Inference/CacheInferModelPhysicalOperator.hpp>
#include <Inference/InferModelPhysicalOperator.hpp>
#include <Inference/InterBufferBatchingPhysicalOperator.hpp>
#include <Inference/PostJoinBatchingPhysicalOperator.hpp>
#include <LoweringRules/AbstractLoweringRule.hpp>
#include <Nautilus/Interface/BufferRef/LowerSchemaProvider.hpp>
#include <Nautilus/Interface/Hash/MurMur3HashFunction.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedEntryMemoryProvider.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Traits/MemoryLayoutTypeTrait.hpp>
#include <Traits/OutputOriginIdsTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <HashMapOptions.hpp>
#include <Inference.hpp>
#include <InferenceConfiguration.hpp>
#include <LoweringRuleRegistry.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>

namespace NES
{

namespace
{

bool containsWindowedAggregation(const LogicalOperator& logicalOperator)
{
    if (logicalOperator.tryGetAs<WindowedAggregationLogicalOperator>().has_value())
    {
        return true;
    }
    return std::ranges::any_of(logicalOperator.getChildren(), [](const auto& child) { return containsWindowedAggregation(child); });
}

bool containsJoin(const LogicalOperator& logicalOperator)
{
    if (logicalOperator.tryGetAs<JoinLogicalOperator>().has_value())
    {
        return true;
    }
    return std::ranges::any_of(logicalOperator.getChildren(), [](const auto& child) { return containsJoin(child); });
}

LogicalOperator getInferenceInputChild(const LogicalOperator& inferModelOperator)
{
    const auto children = inferModelOperator.getChildren();
    PRECONDITION(children.size() == 1, "Expected InferModelLogicalOperator to have exactly one child");

    const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
    if (sequenceOperator.has_value()
        && sequenceOperator.value().get().getSequenceSource() == SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        const auto sequenceChildren = sequenceOperator.value().get().getChildren();
        PRECONDITION(sequenceChildren.size() == 1, "Expected inference SequenceLogicalOperator to have exactly one child");
        return sequenceChildren.front();
    }
    return children.front();
}

uint64_t getInferenceBatchSize(const LogicalOperator& inferModelOperator)
{
    const auto children = inferModelOperator.getChildren();
    if (children.size() != 1)
    {
        return 1;
    }

    const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
    if (!sequenceOperator.has_value()
        || sequenceOperator.value().get().getSequenceSource() != SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        return 1;
    }

    return sequenceOperator.value().get().getBatchSize();
}

std::string makeSyntheticModelInputFieldName(const Schema& inputSchema)
{
    constexpr std::string_view baseName = "__nes_post_join_inference_input";
    std::string candidate{baseName};
    uint64_t suffix = 0;
    while (inputSchema.getFieldByName(candidate).has_value())
    {
        candidate = fmt::format("{}_{}", baseName, ++suffix);
    }
    return candidate;
}

std::string getQualifierPrefix(const std::string& fieldName)
{
    const auto separatorPosition = fieldName.find(Schema::ATTRIBUTE_NAME_SEPARATOR);
    if (separatorPosition == std::string::npos)
    {
        return "";
    }
    return fieldName.substr(0, separatorPosition + std::string_view{Schema::ATTRIBUTE_NAME_SEPARATOR}.size());
}

std::vector<std::string>
makePostJoinOutputFieldNames(const std::vector<std::string>& payloadFieldNames, const std::vector<std::string>& modelOutputFieldNames)
{
    std::vector<std::string> outputFieldNames;
    outputFieldNames.reserve(payloadFieldNames.size() * modelOutputFieldNames.size());
    for (const auto& payloadFieldName : payloadFieldNames)
    {
        const auto qualifierPrefix = getQualifierPrefix(payloadFieldName);
        for (const auto& modelOutputFieldName : modelOutputFieldNames)
        {
            outputFieldNames.push_back(qualifierPrefix + modelOutputFieldName);
        }
    }
    return outputFieldNames;
}

struct PostJoinBatchingInfo
{
    std::vector<std::string> payloadFieldNames;
    std::vector<std::string> outputFieldNames;
    std::string syntheticModelInputFieldName;
    Schema batchInputSchema;
};

std::optional<PostJoinBatchingInfo>
getPostJoinBatchingInfo(const LogicalOperator& logicalOperator, const InferModelLogicalOperator& inferModelOp, const Schema& inputSchema)
{
    if (!inferModelOp.hasVarsizedInput() || inferModelOp.getInputFieldNames().size() <= 1)
    {
        return std::nullopt;
    }

    if (!containsJoin(getInferenceInputChild(logicalOperator)))
    {
        throw UnsupportedQuery(
            "Multiple VARSIZED inference input fields are currently only supported for inference over a window join output");
    }

    for (const auto& payloadFieldName : inferModelOp.getInputFieldNames())
    {
        const auto payloadField = inputSchema.getFieldByName(payloadFieldName);
        if (!payloadField.has_value())
        {
            throw UnsupportedQuery("Post-join inference payload field '{}' does not exist in input schema", payloadFieldName);
        }
        if (payloadField->dataType.type != DataType::Type::VARSIZED || payloadField->dataType.nullable)
        {
            throw UnsupportedQuery("Post-join inference payload field '{}' must be a non-nullable VARSIZED field", payloadFieldName);
        }
    }

    auto batchInputSchema = inputSchema;
    const auto syntheticModelInputFieldName = makeSyntheticModelInputFieldName(batchInputSchema);
    batchInputSchema
        = batchInputSchema.addField(syntheticModelInputFieldName, DataType{DataType::Type::VARSIZED, DataType::NULLABLE::NOT_NULLABLE});

    return PostJoinBatchingInfo{
        .payloadFieldNames = inferModelOp.getInputFieldNames(),
        .outputFieldNames = makePostJoinOutputFieldNames(inferModelOp.getInputFieldNames(), inferModelOp.getOutputFieldNames()),
        .syntheticModelInputFieldName = syntheticModelInputFieldName,
        .batchInputSchema = std::move(batchInputSchema)};
}

InferenceRuntimeOptions getInferenceRuntimeOptions(const QueryExecutionConfiguration& conf)
{
    const auto& inference = conf.inferenceConfiguration;
    return {
        .openvinoInferenceNumThreads = inference.openvinoInferenceNumThreads.getValue(),
        .openvinoNumStreams = inference.openvinoNumStreams.getValue(),
        .openvinoEnableCpuPinning = inference.openvinoEnableCpuPinning.getValue(),
        .openvinoShareCompiledModel = inference.openvinoShareCompiledModel.getValue()};
}

InferenceRuntimeOptions withOpenVinoDynamicBatch(InferenceRuntimeOptions options, const bool enabled)
{
    options.openvinoAllowDynamicBatch = enabled;
    return options;
}

HashMapOptions createBatchDeduplicationHashMapOptions(
    const Schema& inputSchema, const std::vector<std::string>& inputFieldNames, const QueryExecutionConfiguration& conf)
{
    Schema hashMapSchema;
    for (const auto& inputFieldName : inputFieldNames)
    {
        const auto field = inputSchema.getFieldByName(inputFieldName);
        PRECONDITION(field.has_value(), "Expected inference input field {} to exist in schema {}", inputFieldName, inputSchema);
        hashMapSchema.addField(field->name, field->dataType);
    }

    const auto keySize = hashMapSchema.getSizeOfSchemaInBytes();
    constexpr auto valueSize = sizeof(uint64_t) * 2;
    const auto pageSize = conf.pageSize.getValue();
    const auto numberOfBuckets = conf.numberOfPartitions.getValue();
    const auto entrySize = sizeof(ChainedHashMapEntry) + keySize + valueSize;
    const auto entriesPerPage = pageSize / entrySize;
    const auto fieldKeyNames = hashMapSchema.getFieldNames();
    hashMapSchema.addField("rowInputIndex", DataType::Type::UINT64);
    hashMapSchema.addField("rowOutputIndex", DataType::Type::UINT64);

    const auto& [fieldKeys, fieldValues]
        = ChainedEntryMemoryProvider::createFieldOffsets(hashMapSchema, fieldKeyNames, {"rowInputIndex", "rowOutputIndex"});

    return HashMapOptions{
        std::make_unique<MurMur3HashFunction>(),
        {},
        fieldKeys,
        fieldValues,
        entriesPerPage,
        entrySize,
        keySize,
        valueSize,
        pageSize,
        numberOfBuckets};
}

}

LoweringRuleResultSubgraph LowerToPhysicalInferModel::apply(LogicalOperator logicalOperator)
{
    PRECONDITION(logicalOperator.tryGetAs<InferModelLogicalOperator>(), "Expected an InferModelLogicalOperator");
    const auto inferModelOp = logicalOperator.getAs<InferModelLogicalOperator>();

    /// Compile the imported MLIR to IREE bytecode. This is where the
    /// `iree-compile` subprocess runs — deliberately deferred to lowering so
    /// the coordinator only ships the textual MLIR across the wire. The schema
    /// and signature travel with the `ImportedModel` and are propagated by
    /// `compileModel`, so no extra wiring is needed here.
    auto compiled = compileModel(inferModelOp.get().getModel().getImported());
    if (!compiled)
    {
        throw CannotLoadModel("Failed to compile model during lowering: {}", compiled.error().message);
    }
    auto model = std::move(*compiled);

    const auto memoryLayoutTypeTrait = logicalOperator.getTraitSet().tryGet<MemoryLayoutTypeTrait>();
    PRECONDITION(memoryLayoutTypeTrait.has_value(), "Expected a memory layout type trait");
    const auto memoryLayoutType = memoryLayoutTypeTrait.value()->memoryLayout;
    const auto batchSize = getInferenceBatchSize(logicalOperator);
    const auto runtimeOptions = getInferenceRuntimeOptions(conf);
    const auto predictionCacheType = conf.inferenceConfiguration.predictionCacheType.getValue();
    const auto predictionCacheScope = conf.inferenceConfiguration.predictionCacheScope.getValue();
    const auto useBatchDeduplication = conf.inferenceConfiguration.useBatchDeduplication.getValue();
    const auto postJoinBatchingInfo
        = getPostJoinBatchingInfo(logicalOperator, inferModelOp.get(), logicalOperator.getInputSchemas().at(0));

    if (batchSize > 1 || postJoinBatchingInfo.has_value())
    {
        const auto inputSchema
            = postJoinBatchingInfo.has_value() ? postJoinBatchingInfo->batchInputSchema : logicalOperator.getInputSchemas().at(0);
        const auto projections = logicalOperator.getInputSchemas().at(0).getFieldNames();
        const auto inferenceInputFieldNames = postJoinBatchingInfo.has_value()
            ? std::vector<std::string>{postJoinBatchingInfo->syntheticModelInputFieldName}
            : inferModelOp.get().getInputFieldNames();
        const auto inferenceOutputFieldNames
            = postJoinBatchingInfo.has_value() ? postJoinBatchingInfo->outputFieldNames : inferModelOp.get().getOutputFieldNames();
        const auto runtimeBatchSize = postJoinBatchingInfo.has_value() ? postJoinBatchingInfo->payloadFieldNames.size() : batchSize;
        auto bufferRef = LowerSchemaProvider::lowerSchema(conf.operatorBufferSize.getValue(), inputSchema, memoryLayoutType);
        const auto handlerId = getNextOperatorHandlerId();

        const auto outputOriginIdsOpt = getTrait<OutputOriginIdsTrait>(logicalOperator.getTraitSet());
        PRECONDITION(outputOriginIdsOpt.has_value(), "Expected the outputOriginIds trait to be set");
        const auto& outputOriginIds = outputOriginIdsOpt.value().get();
        PRECONDITION(outputOriginIds.size() == 1, "Expected one output origin id");
        auto handler = std::make_shared<BatchInferenceOperatorHandler>(runtimeBatchSize, outputOriginIds[0]);

        PhysicalOperator batchingOperator;
        if (postJoinBatchingInfo.has_value())
        {
            batchingOperator = PostJoinBatchingPhysicalOperator(
                handlerId,
                bufferRef,
                postJoinBatchingInfo->payloadFieldNames,
                postJoinBatchingInfo->outputFieldNames,
                postJoinBatchingInfo->syntheticModelInputFieldName);
        }
        else if (!logicalOperator.getChildren().empty() && containsWindowedAggregation(logicalOperator.getChildren().at(0)))
        {
            batchingOperator = InterBufferBatchingPhysicalOperator(handlerId, bufferRef);
        }
        else
        {
            batchingOperator = BatchingPhysicalOperator(handlerId, bufferRef);
        }

        auto batchingWrapper = std::make_shared<PhysicalOperatorWrapper>(
            batchingOperator,
            inputSchema,
            inputSchema,
            memoryLayoutType,
            memoryLayoutType,
            handlerId,
            handler,
            PhysicalOperatorWrapper::PipelineLocation::EMIT);

        PhysicalOperator physicalOperator;
        if (predictionCacheType != PredictionCacheType::NONE)
        {
            if (predictionCacheScope == PredictionCacheScope::GLOBAL)
            {
                throw UnsupportedQuery(
                    "The global prediction cache is only supported for non-batched inference; use prediction_cache_scope THREAD_LOCAL");
            }
            physicalOperator = BatchCacheInferModelPhysicalOperator(
                std::move(model),
                bufferRef,
                projections,
                inferenceInputFieldNames,
                inferenceOutputFieldNames,
                runtimeBatchSize,
                withOpenVinoDynamicBatch(runtimeOptions, true),
                predictionCacheType,
                conf.inferenceConfiguration.numberOfEntriesPredictionCache.getValue(),
                createBatchDeduplicationHashMapOptions(inputSchema, inferenceInputFieldNames, conf),
                inferModelOp.get().hasVarsizedInput(),
                inferModelOp.get().hasVarsizedOutput(),
                useBatchDeduplication,
                handlerId);

            NES_DEBUG(
                "Lowering InferModel operator to physical BatchCacheInferModelPhysicalOperator operator with batch size {}", batchSize)
        }
        else
        {
            physicalOperator = BatchInferModelPhysicalOperator(
                std::move(model),
                bufferRef,
                projections,
                inferenceInputFieldNames,
                inferenceOutputFieldNames,
                runtimeBatchSize,
                withOpenVinoDynamicBatch(runtimeOptions, useBatchDeduplication),
                createBatchDeduplicationHashMapOptions(inputSchema, inferenceInputFieldNames, conf),
                inferModelOp.get().hasVarsizedInput(),
                inferModelOp.get().hasVarsizedOutput(),
                useBatchDeduplication,
                handlerId);

            NES_DEBUG("Lowering InferModel operator to physical BatchInferModelPhysicalOperator operator with batch size {}", batchSize)
        }

        const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
            physicalOperator,
            logicalOperator.getInputSchemas().at(0),
            logicalOperator.getOutputSchema(),
            memoryLayoutType,
            memoryLayoutType,
            handlerId,
            handler,
            PhysicalOperatorWrapper::PipelineLocation::SCAN,
            std::vector{batchingWrapper});

        std::vector leafes(logicalOperator.getChildren().size(), batchingWrapper);
        return {.root = wrapper, .leafs = {leafes}};
    }

    if (predictionCacheType != PredictionCacheType::NONE)
    {
        auto physicalOperator = CacheInferModelPhysicalOperator(
            std::move(model),
            inferModelOp.get().getInputFieldNames(),
            inferModelOp.get().getOutputFieldNames(),
            runtimeOptions,
            predictionCacheType,
            predictionCacheScope,
            conf.inferenceConfiguration.numberOfEntriesPredictionCache.getValue(),
            inferModelOp.get().hasVarsizedInput(),
            inferModelOp.get().hasVarsizedOutput());

        NES_DEBUG(
            "Lowering InferModel operator to physical CachedInferModelPhysicalOperator operator with {} cache entries",
            conf.inferenceConfiguration.numberOfEntriesPredictionCache.getValue())

        const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
            physicalOperator,
            logicalOperator.getInputSchemas().at(0),
            logicalOperator.getOutputSchema(),
            memoryLayoutType,
            memoryLayoutType,
            PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);

        std::vector leafes(logicalOperator.getChildren().size(), wrapper);
        return {.root = wrapper, .leafs = {leafes}};
    }

    /// Create the physical operator. Input names come from the logical operator: they
    /// were resolved by `withInferredSchema` to the upstream schema's qualified names
    /// (`source$field`), which is what the runtime `record.read` lookup requires.
    /// Output names come from the model — they are the user-declared output field names
    /// that will be written back onto the record.
    auto physicalOperator = InferModelPhysicalOperator(
        std::move(model),
        inferModelOp.get().getInputFieldNames(),
        inferModelOp.get().getOutputFieldNames(),
        runtimeOptions,
        inferModelOp.get().hasVarsizedInput(),
        inferModelOp.get().hasVarsizedOutput());

    NES_DEBUG("Lowering InferModel operator to physical InferModelPhysicalOperator operator")

    const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
        physicalOperator,
        logicalOperator.getInputSchemas().at(0),
        logicalOperator.getOutputSchema(),
        memoryLayoutType,
        memoryLayoutType,
        PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);

    std::vector leafes(logicalOperator.getChildren().size(), wrapper);
    return {.root = wrapper, .leafs = {leafes}};
}

std::unique_ptr<AbstractLoweringRule>
LoweringRuleGeneratedRegistrar::RegisterInferModelLoweringRule(LoweringRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalInferModel>(argument.conf);
}

}
