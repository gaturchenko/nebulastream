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

#include <BatchInferModelPhysicalOperator.hpp>
#include <BatchInferenceOperatorHandler.hpp>
#include <BatchingPhysicalOperator.hpp>
#include <EmitOperatorHandler.hpp>
#include <EmitPhysicalOperator.hpp>
#include <Inference.hpp>
#include <InterBufferBatchingPhysicalOperator.hpp>
#include <Model.hpp>
#include <PerfEvent.hpp>
#include <PhysicalOperator.hpp>
#include <ScanPhysicalOperator.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Identifiers/NESStrongType.hpp>
#include <Nautilus/Interface/BufferRef/LowerSchemaProvider.hpp>
#include <Pipelines/CompiledExecutablePipelineStage.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <folly/Synchronized.h>
#include <ErrorHandling.hpp>
#include <Pipeline.hpp>
#include <PipelineExecutionContext.hpp>
#include <TestTupleBuffer.hpp>
#include <options.hpp>

namespace NES
{

namespace
{

constexpr size_t bufferSize = 8192;
constexpr size_t numFloats = 100;
constexpr uint64_t inputBufferPoolSize = 1;

struct MockedPipelineContext final : PipelineExecutionContext
{
    using EmitCallback = std::function<void(const TupleBuffer&)>;

    bool emitBuffer(const TupleBuffer& buffer, ContinuationPolicy) override
    {
        if (emitCallback)
        {
            emitCallback(buffer);
        }
        else
        {
            buffers.wlock()->emplace_back(buffer);
        }
        if (releasePinnedBuffersOnEmit)
        {
            pinnedBuffers.clear();
        }
        return true;
    }

    TupleBuffer allocateTupleBuffer() override { return bufferManager->getBufferBlocking(); }

    [[nodiscard]] WorkerThreadId getWorkerThreadId() const override { return INITIAL<WorkerThreadId>; }

    [[nodiscard]] uint64_t getNumberOfWorkerThreads() const override { return 1; }

    [[nodiscard]] std::shared_ptr<AbstractBufferProvider> getBufferManager() const override { return bufferManager; }

    [[nodiscard]] PipelineId getPipelineId() const override { return PipelineId(1); }

    std::unordered_map<OperatorHandlerId, std::shared_ptr<OperatorHandler>>& getOperatorHandlers() override { return *operatorHandlers; }

    void setOperatorHandlers(std::unordered_map<OperatorHandlerId, std::shared_ptr<OperatorHandler>>& opHandlers) override
    {
        operatorHandlers = &opHandlers;
    }

    void repeatTask(const TupleBuffer&, std::chrono::milliseconds) override
    {
        throw std::runtime_error("repeatTask should not be called by the benchmark pipeline");
    }

    TupleBuffer& pinBuffer(TupleBuffer&& tupleBuffer) override
    {
        pinnedBuffers.emplace_back(std::make_unique<TupleBuffer>(std::move(tupleBuffer)));
        return *pinnedBuffers.back();
    }

    ///NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members) lifetime is controlled by the benchmark iteration
    folly::Synchronized<std::vector<TupleBuffer>>& buffers;
    std::shared_ptr<BufferManager> bufferManager;
    EmitCallback emitCallback;
    bool releasePinnedBuffersOnEmit = false;
    std::unordered_map<OperatorHandlerId, std::shared_ptr<OperatorHandler>>* operatorHandlers = nullptr;
    std::vector<std::unique_ptr<TupleBuffer>> pinnedBuffers;

    MockedPipelineContext(
        folly::Synchronized<std::vector<TupleBuffer>>& buffers,
        std::shared_ptr<BufferManager> bufferManager,
        EmitCallback emitCallback = {},
        const bool releasePinnedBuffersOnEmit = false)
        : buffers(buffers)
        , bufferManager(std::move(bufferManager))
        , emitCallback(std::move(emitCallback))
        , releasePinnedBuffersOnEmit(releasePinnedBuffersOnEmit)
    {
    }
};

struct LoweredBatchInferencePipelines
{
    std::shared_ptr<Pipeline> batchingPipeline;
    std::shared_ptr<Pipeline> inferencePipeline;
    std::unordered_map<OperatorHandlerId, std::shared_ptr<OperatorHandler>> handlers;
    uint64_t inputRecordsPerBuffer;
    uint64_t outputRecordsPerBuffer;
};

struct BenchmarkConfig
{
    std::filesystem::path modelsDir = std::filesystem::path(INFERENCE_TEST_DATA);
    std::filesystem::path outputFile = std::filesystem::path("BatchInferenceMicrobenchmark.csv");
    uint64_t records = 16;
    uint64_t repetitions = 10;
    uint64_t warmups = 2;
    std::vector<size_t> batchSizes = {1, 2, 4, 8, 16};
    bool runIree = false;
    bool runOpenVino = true;
};

struct BenchmarkModel
{
    std::string modelName;
    std::string_view backendName;
    std::optional<CompiledModel> model;
};

struct Measurement
{
    uint64_t run;
    double durationUs;
    double throughputRecordsPerSecond;
    double cyclesPerRecord;
    double instructionsPerRecord;
    double cyclesPerBatch;
    double instructionsPerBatch;
    double l1MissesPerRecord;
    double llcMissesPerRecord;
    double cacheMissesPerKInstr;
    double branchMissRate;
    double dtlbMissesPerRecord;
    double ipc;
    double ghz;
};

struct PerfCounters
{
    double durationSeconds = 0;
    double cycles = 0;
    double instructions = 0;
    double l1Misses = 0;
    double llcMisses = 0;
    double cacheMisses = 0;
    double branchInstructions = 0;
    double branchMisses = 0;
    double dtlbMisses = 0;
    double taskClock = 0;
};

struct RunResult
{
    uint64_t emittedRecords;
    PerfCounters perfCounters;
};

std::vector<float> makeFloats(const size_t count, const float startValue)
{
    std::vector<float> vec(count);
    /// NOLINTNEXTLINE(modernize-use-ranges) std::ranges::iota not yet available in libc++
    std::iota(vec.begin(), vec.end(), startValue);
    return vec;
}

std::vector<std::vector<float>> makeInputRecords(const uint64_t records)
{
    std::vector<std::vector<float>> inputs;
    inputs.reserve(records);
    for (uint64_t row = 0; row < records; ++row)
    {
        inputs.push_back(makeFloats(numFloats, static_cast<float>((row * numFloats) + 1)));
    }
    return inputs;
}

std::vector<std::string> makeOutputFieldNames(const size_t outputFields)
{
    std::vector<std::string> names;
    names.reserve(outputFields);
    for (size_t i = 0; i < outputFields; ++i)
    {
        names.push_back("out_" + std::to_string(i));
    }
    return names;
}

std::pair<Schema, Schema> makeSchemas(const std::vector<std::string>& outputFieldNames)
{
    Schema inputSchema;
    inputSchema.addField("input_blob", DataType::Type::VARSIZED);

    Schema outputSchema;
    outputSchema.addField("input_blob", DataType::Type::VARSIZED);
    for (const auto& name : outputFieldNames)
    {
        outputSchema.addField(name, DataType::Type::FLOAT32);
    }
    return {inputSchema, outputSchema};
}

std::vector<TupleBuffer> createInputBuffers(const Schema& inputSchema, const std::vector<std::vector<float>>& recordFloats)
{
    const auto tupleSize = inputSchema.getSizeOfSchemaInBytes();
    if (tupleSize == 0)
    {
        throw std::invalid_argument("input schema tuple size must be greater than zero");
    }
    const auto recordsPerInputBuffer = bufferSize / tupleSize;
    if (recordsPerInputBuffer == 0)
    {
        throw std::invalid_argument("input schema tuple size exceeds the benchmark tuple buffer size");
    }

    auto bufMgr = BufferManager::create(bufferSize, inputBufferPoolSize);

    uint64_t payloadSize = 0;
    for (const auto& floats : recordFloats)
    {
        payloadSize += floats.size() * sizeof(float);
    }

    auto childBuffer = bufMgr->getUnpooledBuffer(std::max<uint64_t>(payloadSize, 1));
    if (!childBuffer.has_value())
    {
        throw std::runtime_error("Failed to allocate input child buffer of size " + std::to_string(payloadSize));
    }
    childBuffer->setNumberOfTuples(payloadSize);

    auto childMemory = childBuffer->getAvailableMemoryArea();
    uint64_t payloadOffset = 0;
    uint64_t recordIndex = 0;
    std::vector<TupleBuffer> tupleBuffers;
    tupleBuffers.reserve((recordFloats.size() + recordsPerInputBuffer - 1) / recordsPerInputBuffer);
    while (recordIndex < recordFloats.size())
    {
        auto tupleBuffer = bufMgr->getUnpooledBuffer(bufferSize);
        if (!tupleBuffer.has_value())
        {
            throw std::runtime_error("Failed to allocate input tuple buffer");
        }
        tupleBuffer->setSequenceNumber(SequenceNumber(1));
        tupleBuffer->setChunkNumber(ChunkNumber(tupleBuffers.size() + 1));
        tupleBuffer->setLastChunk(recordIndex + recordsPerInputBuffer >= recordFloats.size());
        tupleBuffer->setOriginId(INITIAL<OriginId>);

        auto parentRecords = tupleBuffer->getAvailableMemoryArea<VariableSizedAccess>();
        const auto recordsInBuffer = std::min<uint64_t>(recordsPerInputBuffer, recordFloats.size() - recordIndex);
        for (uint64_t localRecord = 0; localRecord < recordsInBuffer; ++localRecord)
        {
            const auto& floats = recordFloats[recordIndex + localRecord];
            const auto recordPayloadSize = floats.size() * sizeof(float);
            /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) float-to-byte packing
            const auto* rawData = reinterpret_cast<const std::byte*>(floats.data());
            std::memcpy(childMemory.data() + payloadOffset, rawData, recordPayloadSize);
            parentRecords[localRecord] = VariableSizedAccess{
                VariableSizedAccess::Index{0},
                VariableSizedAccess::Offset{payloadOffset},
                VariableSizedAccess::Size{recordPayloadSize}};
            payloadOffset += recordPayloadSize;
        }
        tupleBuffer->setNumberOfTuples(recordsInBuffer);
        auto childBufferRef = *childBuffer;
        const auto childIndex = tupleBuffer->storeChildBuffer(childBufferRef);
        INVARIANT(childIndex == VariableSizedAccess::Index{0}, "Synthetic input buffer should have exactly one child buffer");

        tupleBuffers.emplace_back(std::move(*tupleBuffer));
        recordIndex += recordsInBuffer;
    }
    return tupleBuffers;
}

uint64_t divCeil(const uint64_t value, const uint64_t divisor)
{
    if (divisor == 0)
    {
        throw std::invalid_argument("divisor must be greater than zero");
    }
    return (value + divisor - 1) / divisor;
}

uint64_t getNumberOfBatches(const uint64_t records, const uint64_t batchSize)
{
    return divCeil(records, batchSize);
}

uint64_t getOutputBufferAllocations(const uint64_t records, const uint64_t batchSize, const uint64_t outputRecordsPerBuffer)
{
    const auto fullBatches = records / batchSize;
    const auto tailRecords = records % batchSize;
    auto outputBuffers = fullBatches * divCeil(batchSize, outputRecordsPerBuffer);
    if (tailRecords > 0)
    {
        outputBuffers += divCeil(tailRecords, outputRecordsPerBuffer);
    }
    return outputBuffers;
}

uint64_t getVarSizedChildBufferAllocationsForChunk(uint64_t records, const uint64_t recordsPerFixedBuffer, const uint64_t bytesPerRecord)
{
    if (records == 0 || bytesPerRecord >= bufferSize)
    {
        return 0;
    }

    /// TupleBufferRef allocates a new child buffer when used bytes plus the new value is >= buffer size.
    const auto recordsPerChildBuffer = (bufferSize - 1) / bytesPerRecord;
    uint64_t childBuffers = 0;
    while (records > 0)
    {
        const auto pageRecords = std::min(records, recordsPerFixedBuffer);
        childBuffers += divCeil(pageRecords, recordsPerChildBuffer);
        records -= pageRecords;
    }
    return childBuffers;
}

uint64_t getVarSizedChildBufferAllocations(
    const uint64_t records, const uint64_t batchSize, const uint64_t recordsPerFixedBuffer, const uint64_t bytesPerRecord)
{
    const auto fullBatches = records / batchSize;
    const auto tailRecords = records % batchSize;
    auto childBuffers = fullBatches * getVarSizedChildBufferAllocationsForChunk(batchSize, recordsPerFixedBuffer, bytesPerRecord);
    if (tailRecords > 0)
    {
        childBuffers += getVarSizedChildBufferAllocationsForChunk(tailRecords, recordsPerFixedBuffer, bytesPerRecord);
    }
    return childBuffers;
}

uint64_t getRuntimeBufferPoolSize(
    const uint64_t records, const uint64_t batchSize, const uint64_t inputRecordsPerBuffer, const uint64_t outputRecordsPerBuffer)
{
    const auto maxLiveInputRecords = std::min(records, inputRecordsPerBuffer + batchSize - 1);
    const auto maxLiveOutputRecords = std::min(records, batchSize);
    const auto inputBytesPerRecord = numFloats * sizeof(float);
    const auto emittedBatchBuffers = getNumberOfBatches(maxLiveInputRecords, batchSize);
    constexpr uint64_t safetyMargin = 8;
    const auto outputBuffers = getOutputBufferAllocations(maxLiveOutputRecords, maxLiveOutputRecords, outputRecordsPerBuffer);
    const auto batchChildBuffers
        = getVarSizedChildBufferAllocations(maxLiveInputRecords, batchSize, inputRecordsPerBuffer, inputBytesPerRecord);
    const auto outputChildBuffers
        = getVarSizedChildBufferAllocations(maxLiveOutputRecords, maxLiveOutputRecords, outputRecordsPerBuffer, inputBytesPerRecord);

    return emittedBatchBuffers + outputBuffers + batchChildBuffers + outputChildBuffers + safetyMargin;
}

LoweredBatchInferencePipelines createLoweredBatchInferencePipelines(
    const CompiledModel& model,
    const Schema& inputSchema,
    const Schema& outputSchema,
    const std::vector<std::string>& inputFieldNames,
    const std::vector<std::string>& outputFieldNames,
    const size_t batchSize)
{
    auto inputBufRef = LowerSchemaProvider::lowerSchema(bufferSize, inputSchema, MemoryLayoutType::ROW_LAYOUT);
    auto outputBufRef = LowerSchemaProvider::lowerSchema(bufferSize, outputSchema, MemoryLayoutType::ROW_LAYOUT);
    const OperatorHandlerId batchHandlerId(1);
    const OperatorHandlerId emitHandlerId(2);

    ScanPhysicalOperator scan(inputBufRef, inputSchema.getFieldNames());
    InterBufferBatchingPhysicalOperator batching(batchHandlerId, inputBufRef);
    scan.setChild(PhysicalOperator(batching));
    auto batchingPipeline = std::make_shared<Pipeline>(PhysicalOperator(scan));

    BatchInferModelPhysicalOperator batchInferModel(
        model, inputBufRef, inputSchema.getFieldNames(), inputFieldNames, outputFieldNames, batchSize, true, false, batchHandlerId);
    const EmitPhysicalOperator emit(emitHandlerId, outputBufRef);
    batchInferModel.setChild(PhysicalOperator(emit));
    auto inferencePipeline = std::make_shared<Pipeline>(PhysicalOperator(batchInferModel));

    std::unordered_map<OperatorHandlerId, std::shared_ptr<OperatorHandler>> handlers;
    handlers[batchHandlerId] = std::make_shared<BatchInferenceOperatorHandler>(batchSize, INITIAL<OriginId>);
    handlers[emitHandlerId] = std::make_shared<EmitOperatorHandler>();

    return {
        .batchingPipeline = std::move(batchingPipeline),
        .inferencePipeline = std::move(inferencePipeline),
        .handlers = std::move(handlers),
        .inputRecordsPerBuffer = inputBufRef->getCapacity(),
        .outputRecordsPerBuffer = outputBufRef->getCapacity()};
}

std::expected<CompiledModel, std::string> importAndCompile(const std::string& path, const ModelBackend backend)
{
    auto imported = importModel(path, backend);
    if (!imported)
    {
        return std::unexpected(imported.error().message);
    }
    auto compiled = compileModel(*imported);
    if (!compiled)
    {
        return std::unexpected(compiled.error().message);
    }
    return std::move(*compiled);
}

std::string lowerExtension(const std::filesystem::path& path)
{
    auto extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) { return std::tolower(character); });
    return extension;
}

std::vector<std::filesystem::path> discoverOnnxModels(const std::filesystem::path& modelsDir)
{
    if (!std::filesystem::exists(modelsDir))
    {
        throw std::invalid_argument("models_dir does not exist: " + modelsDir.string());
    }
    if (!std::filesystem::is_directory(modelsDir))
    {
        throw std::invalid_argument("models_dir is not a directory: " + modelsDir.string());
    }

    std::vector<std::filesystem::path> models;
    for (const auto& entry : std::filesystem::directory_iterator(modelsDir))
    {
        if (entry.is_regular_file() && lowerExtension(entry.path()) == ".onnx")
        {
            models.emplace_back(entry.path());
        }
    }
    std::ranges::sort(models);

    if (models.empty())
    {
        throw std::invalid_argument("models_dir must contain at least one ONNX model: " + modelsDir.string());
    }
    return models;
}

std::string csvEscape(const std::string_view value)
{
    if (value.find_first_of(",\"\n") == std::string_view::npos)
    {
        return std::string(value);
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const auto character : value)
    {
        if (character == '"')
        {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

double missingMetric()
{
    return std::numeric_limits<double>::quiet_NaN();
}

double safeDivide(const double numerator, const double denominator)
{
    if (numerator < 0 || denominator <= 0)
    {
        return missingMetric();
    }
    return numerator / denominator;
}

void writeMetric(std::ostream& csv, const double value)
{
    if (std::isnan(value))
    {
        return;
    }
    csv << value;
}

struct PerfMetrics
{
    double cyclesPerRecord;
    double instructionsPerRecord;
    double cyclesPerBatch;
    double instructionsPerBatch;
    double l1MissesPerRecord;
    double llcMissesPerRecord;
    double cacheMissesPerKInstr;
    double branchMissRate;
    double dtlbMissesPerRecord;
    double ipc;
    double ghz;
};

void addPerfWindow(PerfCounters& counters, PerfEvent& perfEvent)
{
    counters.durationSeconds += perfEvent.getDuration();
    counters.cycles += perfEvent.getCounter("cycle");
    counters.instructions += perfEvent.getCounter("instr");
    counters.l1Misses += perfEvent.getCounter("L1-miss");
    counters.llcMisses += perfEvent.getCounter("LLC-miss");
    counters.cacheMisses += perfEvent.getCounter("cache-miss");
    counters.branchInstructions += perfEvent.getCounter("branch");
    counters.branchMisses += perfEvent.getCounter("br-miss");
    counters.dtlbMisses += perfEvent.getCounter("dTLB-miss");
    counters.taskClock += perfEvent.getCounter("task");
}

PerfMetrics computePerfMetrics(const PerfCounters& counters, const uint64_t emittedRecords, const uint64_t batches)
{
    return {
        .cyclesPerRecord = safeDivide(counters.cycles, static_cast<double>(emittedRecords)),
        .instructionsPerRecord = safeDivide(counters.instructions, static_cast<double>(emittedRecords)),
        .cyclesPerBatch = safeDivide(counters.cycles, static_cast<double>(batches)),
        .instructionsPerBatch = safeDivide(counters.instructions, static_cast<double>(batches)),
        .l1MissesPerRecord = safeDivide(counters.l1Misses, static_cast<double>(emittedRecords)),
        .llcMissesPerRecord = safeDivide(counters.llcMisses, static_cast<double>(emittedRecords)),
        .cacheMissesPerKInstr = safeDivide(counters.cacheMisses, counters.instructions / 1'000.0),
        .branchMissRate = safeDivide(counters.branchMisses, counters.branchInstructions),
        .dtlbMissesPerRecord = safeDivide(counters.dtlbMisses, static_cast<double>(emittedRecords)),
        .ipc = safeDivide(counters.instructions, counters.cycles),
        .ghz = safeDivide(counters.cycles, counters.taskClock)};
}

void writePerfMetrics(std::ostream& csv, const Measurement& measurement)
{
    csv << ',';
    writeMetric(csv, measurement.cyclesPerRecord);
    csv << ',';
    writeMetric(csv, measurement.instructionsPerRecord);
    csv << ',';
    writeMetric(csv, measurement.cyclesPerBatch);
    csv << ',';
    writeMetric(csv, measurement.instructionsPerBatch);
    csv << ',';
    writeMetric(csv, measurement.l1MissesPerRecord);
    csv << ',';
    writeMetric(csv, measurement.llcMissesPerRecord);
    csv << ',';
    writeMetric(csv, measurement.cacheMissesPerKInstr);
    csv << ',';
    writeMetric(csv, measurement.branchMissRate);
    csv << ',';
    writeMetric(csv, measurement.dtlbMissesPerRecord);
    csv << ',';
    writeMetric(csv, measurement.ipc);
    csv << ',';
    writeMetric(csv, measurement.ghz);
}

void writeSkippedRow(std::ostream& csv, const std::string& modelName, const std::string_view backendName)
{
    csv << csvEscape(modelName) << ',' << backendName << ",,,model_unavailable";
    for (uint64_t i = 0; i < 19; ++i)
    {
        csv << ',';
    }
    csv << '\n';
}

size_t getOutputFieldCount(const CompiledModel& model)
{
    if (model.outputSize() == 0 || model.outputSize() % sizeof(float) != 0)
    {
        throw std::invalid_argument("model output size must be a non-zero multiple of sizeof(float)");
    }
    return model.outputSize() / sizeof(float);
}

std::vector<size_t> parseBatchSizes(std::string_view raw)
{
    std::vector<size_t> batchSizes;
    std::stringstream stream{std::string(raw)};
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            continue;
        }

        const auto batchSize = std::stoull(item);
        if (batchSize > 0)
        {
            batchSizes.push_back(batchSize);
        }
    }
    return batchSizes;
}

uint64_t readEnvUInt(const char* name, const uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || std::string_view(value).empty())
    {
        return defaultValue;
    }
    return std::stoull(value);
}

std::vector<size_t> readEnvBatchSizes()
{
    const char* value = std::getenv("NES_BATCH_INFERENCE_MICROBENCH_BATCH_SIZES");
    if (value == nullptr || std::string_view(value).empty())
    {
        return {1, 2, 4, 8, 16};
    }
    return parseBatchSizes(value);
}

void printUsage(const char* binary)
{
    std::cout << "Usage: " << binary
              << " [--models-dir DIR] [--output FILE] [--records N] [--repetitions N] [--warmups N] [--batch-sizes 1,2,4]"
                 " [--backend all|iree|openvino]\n";
}

BenchmarkConfig parseConfig(const int argc, char** argv)
{
    BenchmarkConfig config{
        .modelsDir = std::filesystem::path(
            std::getenv("NES_BATCH_INFERENCE_MICROBENCH_MODELS_DIR") != nullptr ? std::getenv("NES_BATCH_INFERENCE_MICROBENCH_MODELS_DIR")
                                                                                : INFERENCE_TEST_DATA),
        .outputFile = std::filesystem::path(
            std::getenv("NES_BATCH_INFERENCE_MICROBENCH_OUTPUT") != nullptr ? std::getenv("NES_BATCH_INFERENCE_MICROBENCH_OUTPUT")
                                                                            : "BatchInferenceMicrobenchmark.csv"),
        .records = readEnvUInt("NES_BATCH_INFERENCE_MICROBENCH_RECORDS", 16),
        .repetitions = readEnvUInt("NES_BATCH_INFERENCE_MICROBENCH_REPETITIONS", 10),
        .warmups = readEnvUInt("NES_BATCH_INFERENCE_MICROBENCH_WARMUPS", 2),
        .batchSizes = readEnvBatchSizes(),
        .runIree = false,
        .runOpenVino = true};

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        const auto nextValue = [&](const std::string_view option) -> std::string_view
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for " + std::string(option));
            }
            ++i;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--models-dir" || arg == "--models_dir")
        {
            config.modelsDir = std::filesystem::path(std::string(nextValue(arg)));
        }
        else if (arg == "--output" || arg == "--output-file" || arg == "--output_file")
        {
            config.outputFile = std::filesystem::path(std::string(nextValue(arg)));
        }
        else if (arg == "--records")
        {
            config.records = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--repetitions")
        {
            config.repetitions = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--warmups")
        {
            config.warmups = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--batch-sizes")
        {
            config.batchSizes = parseBatchSizes(nextValue(arg));
        }
        else if (arg == "--backend")
        {
            const auto backend = nextValue(arg);
            if (backend == "all")
            {
                config.runIree = true;
                config.runOpenVino = true;
            }
            else if (backend == "iree")
            {
                config.runIree = true;
                config.runOpenVino = false;
            }
            else if (backend == "openvino")
            {
                config.runIree = false;
                config.runOpenVino = true;
            }
            else
            {
                throw std::invalid_argument("Unknown backend: " + std::string(backend));
            }
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (config.records == 0)
    {
        throw std::invalid_argument("records must be greater than zero");
    }
    if (config.repetitions == 0)
    {
        throw std::invalid_argument("repetitions must be greater than zero");
    }
    if (config.batchSizes.empty())
    {
        throw std::invalid_argument("at least one batch size must be configured");
    }
    if (!config.runIree && !config.runOpenVino)
    {
        throw std::invalid_argument("at least one backend must be enabled");
    }

    return config;
}

RunResult runOnce(
    CompiledExecutablePipelineStage& batchingStage,
    CompiledExecutablePipelineStage& inferenceStage,
    BatchInferenceOperatorHandler& batchHandler,
    const std::vector<TupleBuffer>& inputBuffers,
    const uint64_t runtimeBufferPoolSize,
    const SequenceNumber sequenceNumber,
    PerfEvent& perfEvent)
{
    folly::Synchronized<std::vector<TupleBuffer>> emittedBatchBuffers;
    folly::Synchronized<std::vector<TupleBuffer>> outputBuffers;
    auto bufMgr = BufferManager::create(bufferSize, runtimeBufferPoolSize);
    uint64_t emittedRecords = 0;
    PerfCounters perfCounters;
    MockedPipelineContext inferencePec{
        outputBuffers,
        bufMgr,
        [&emittedRecords](const TupleBuffer& outputBuffer) { emittedRecords += outputBuffer.getNumberOfTuples(); },
        true};
    MockedPipelineContext batchingPec{
        emittedBatchBuffers,
        bufMgr};

    const auto drainBatchBuffers = [&]()
    {
        std::vector<TupleBuffer> batchBuffers;
        {
            auto lockedBatchBuffers = emittedBatchBuffers.wlock();
            batchBuffers.swap(*lockedBatchBuffers);
        }
        if (batchBuffers.empty())
        {
            return;
        }

        perfEvent.startCounters();
        for (auto& batchBuffer : batchBuffers)
        {
            inferenceStage.execute(batchBuffer, inferencePec);
        }
        perfEvent.stopCounters();
        addPerfWindow(perfCounters, perfEvent);
        batchHandler.garbageCollectBatches();
    };

    batchingStage.start(batchingPec);
    inferenceStage.start(inferencePec);
    for (const auto& inputBuffer : inputBuffers)
    {
        auto currentInputBuffer = inputBuffer;
        currentInputBuffer.setSequenceNumber(sequenceNumber);
        batchingStage.execute(currentInputBuffer, batchingPec);
        drainBatchBuffers();
    }

    batchingStage.stop(batchingPec);
    drainBatchBuffers();
    inferenceStage.stop(inferencePec);

    return {.emittedRecords = emittedRecords, .perfCounters = perfCounters};
}

void runBenchmarkForModel(
    const BenchmarkModel& benchmarkModel,
    const BenchmarkConfig& config,
    const Schema& inputSchema,
    const std::vector<TupleBuffer>& inputBuffers,
    std::ostream& csv)
{
    if (!benchmarkModel.model.has_value())
    {
        writeSkippedRow(csv, benchmarkModel.modelName, benchmarkModel.backendName);
        return;
    }

    const auto outputFieldNames = makeOutputFieldNames(getOutputFieldCount(*benchmarkModel.model));
    const auto outputSchema = makeSchemas(outputFieldNames).second;

    for (const auto batchSize : config.batchSizes)
    {
        auto [batchingPipeline, inferencePipeline, handlers, inputRecordsPerBuffer, outputRecordsPerBuffer]
            = createLoweredBatchInferencePipelines(
                *benchmarkModel.model, inputSchema, outputSchema, {"input_blob"}, outputFieldNames, batchSize);
        const auto batchHandler = std::dynamic_pointer_cast<BatchInferenceOperatorHandler>(handlers.at(OperatorHandlerId(1)));
        PRECONDITION(batchHandler != nullptr, "Batch inference benchmark expected a BatchInferenceOperatorHandler");
        const auto runtimeBufferPoolSize
            = getRuntimeBufferPoolSize(config.records, batchSize, inputRecordsPerBuffer, outputRecordsPerBuffer);

        nautilus::engine::Options options;
        options.setOption("engine.Compilation", true);
        CompiledExecutablePipelineStage batchingStage(batchingPipeline, handlers, options);
        CompiledExecutablePipelineStage inferenceStage(inferencePipeline, handlers, options);

        std::vector<Measurement> measurements;
        measurements.reserve(config.repetitions);
        uint64_t emittedRecords = 0;
        PerfEvent perfEvent;

        for (uint64_t iteration = 0; iteration < config.warmups + config.repetitions; ++iteration)
        {
            const auto runResult = runOnce(
                batchingStage,
                inferenceStage,
                *batchHandler,
                inputBuffers,
                runtimeBufferPoolSize,
                SequenceNumber(iteration + 1),
                perfEvent);
            emittedRecords = runResult.emittedRecords;

            const auto duration = runResult.perfCounters.durationSeconds;
            const auto durationUs = duration * 1'000'000.0;

            if (emittedRecords != config.records)
            {
                throw std::runtime_error(
                    "Expected " + std::to_string(config.records) + " emitted records, got " + std::to_string(emittedRecords));
            }

            if (iteration >= config.warmups)
            {
                const auto perfMetrics
                    = computePerfMetrics(runResult.perfCounters, emittedRecords, getNumberOfBatches(emittedRecords, batchSize));
                measurements.push_back(Measurement{
                    .run = iteration - config.warmups,
                    .durationUs = durationUs,
                    .throughputRecordsPerSecond = static_cast<double>(emittedRecords) / duration,
                    .cyclesPerRecord = perfMetrics.cyclesPerRecord,
                    .instructionsPerRecord = perfMetrics.instructionsPerRecord,
                    .cyclesPerBatch = perfMetrics.cyclesPerBatch,
                    .instructionsPerBatch = perfMetrics.instructionsPerBatch,
                    .l1MissesPerRecord = perfMetrics.l1MissesPerRecord,
                    .llcMissesPerRecord = perfMetrics.llcMissesPerRecord,
                    .cacheMissesPerKInstr = perfMetrics.cacheMissesPerKInstr,
                    .branchMissRate = perfMetrics.branchMissRate,
                    .dtlbMissesPerRecord = perfMetrics.dtlbMissesPerRecord,
                    .ipc = perfMetrics.ipc,
                    .ghz = perfMetrics.ghz});
            }
        }

        const auto [minIt, maxIt] = std::minmax_element(
            measurements.begin(),
            measurements.end(),
            [](const auto& left, const auto& right) { return left.durationUs < right.durationUs; });
        const auto totalUs = std::accumulate(
            measurements.begin(),
            measurements.end(),
            0.0,
            [](const auto sum, const auto& measurement) { return sum + measurement.durationUs; });
        const auto avgUs = totalUs / static_cast<double>(measurements.size());
        const auto avgRecordsPerSecond = static_cast<double>(config.records) * 1'000'000.0 / avgUs;
        for (const auto& measurement : measurements)
        {
            csv << csvEscape(benchmarkModel.modelName) << ',' << benchmarkModel.backendName << ',' << batchSize << ',' << measurement.run
                << ",ok," << measurement.durationUs << ',' << measurement.throughputRecordsPerSecond << ',' << avgUs << ','
                << minIt->durationUs << ',' << maxIt->durationUs << ',' << avgRecordsPerSecond << ',' << emittedRecords << ','
                << runtimeBufferPoolSize;
            writePerfMetrics(csv, measurement);
            csv << '\n';
        }
    }
}

int runBatchInferenceMicrobenchmark(const BenchmarkConfig& config)
{
    const auto modelPaths = discoverOnnxModels(config.modelsDir);
    const auto inputSchema = makeSchemas({}).first;
    auto inputBuffers = createInputBuffers(inputSchema, makeInputRecords(config.records));

    std::ofstream csv(config.outputFile);
    if (!csv.is_open())
    {
        throw std::runtime_error("Failed to open benchmark output CSV: " + config.outputFile.string());
    }

    std::cerr << "Batch inference microbenchmark"
              << " models_dir=" << config.modelsDir << " models=" << modelPaths.size() << " records=" << config.records
              << " tuple_buffer_size=" << bufferSize << " repetitions=" << config.repetitions
              << " input_buffer_pool_size=" << inputBufferPoolSize << " warmups=" << config.warmups << " output=" << config.outputFile
              << '\n';
    csv << "model,backend,batch_size,run,status,duration_us,throughput_records_per_second,avg_us,min_us,max_us,"
           "avg_records_per_second,emitted_records,runtime_buffer_pool_size,cycles_per_record,instructions_per_record,"
           "cycles_per_batch,instructions_per_batch,l1_misses_per_record,llc_misses_per_record,cache_misses_per_kinstr,"
           "branch_miss_rate,dtlb_misses_per_record,ipc,ghz\n";

    for (const auto& modelPath : modelPaths)
    {
        const auto modelName = modelPath.filename().string();

        std::vector<BenchmarkModel> benchmarkModels;
        if (config.runIree)
        {
            auto ireeModel = importAndCompile(modelPath.string(), ModelBackend::IREE);
            if (ireeModel.has_value())
            {
                benchmarkModels.push_back(BenchmarkModel{.modelName = modelName, .backendName = "IREE", .model = std::move(*ireeModel)});
            }
            else
            {
                std::cerr << "IREE model unavailable for " << modelPath << ": " << ireeModel.error() << '\n';
                benchmarkModels.push_back(BenchmarkModel{.modelName = modelName, .backendName = "IREE", .model = std::nullopt});
            }
        }
        if (config.runOpenVino)
        {
            auto openVinoModel = importAndCompile(modelPath.string(), ModelBackend::OPENVINO);
            if (openVinoModel.has_value())
            {
                benchmarkModels.push_back(
                    BenchmarkModel{.modelName = modelName, .backendName = "OpenVINO", .model = std::move(*openVinoModel)});
            }
            else
            {
                std::cerr << "OpenVINO model unavailable for " << modelPath << ": " << openVinoModel.error() << '\n';
                benchmarkModels.push_back(BenchmarkModel{.modelName = modelName, .backendName = "OpenVINO", .model = std::nullopt});
            }
        }

        for (const auto& benchmarkModel : benchmarkModels)
        {
            runBenchmarkForModel(benchmarkModel, config, inputSchema, inputBuffers, csv);
        }
    }

    csv.flush();
    if (!csv.good())
    {
        throw std::runtime_error("Failed while writing benchmark output CSV: " + config.outputFile.string());
    }
    std::cerr << "Wrote benchmark CSV to " << config.outputFile << '\n';

    return EXIT_SUCCESS;
}

}

}

int main(int argc, char** argv)
{
    try
    {
        NES::Logger::setupLogging("BatchInferenceMicrobenchmark.log", NES::LogLevel::LOG_ERROR);
        return NES::runBatchInferenceMicrobenchmark(NES::parseConfig(argc, argv));
    }
    catch (const std::exception& ex)
    {
        std::cerr << "BatchInferenceMicrobenchmark failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
