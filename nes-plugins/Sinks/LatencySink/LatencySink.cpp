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

#include <LatencySink.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <Configurations/Descriptor.hpp>
#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sinks/Sink.hpp>
#include <Sinks/SinkDescriptor.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <PipelineExecutionContext.hpp>
#include <SinkRegistry.hpp>
#include <SinkValidationRegistry.hpp>

namespace NES
{

namespace
{
/// Microseconds since the Unix epoch on the wall clock; the same clock CURRENT_TIME() stamps with, so
/// `recv - ingest` is a meaningful single-node latency.
std::uint64_t nowMicros()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string toUpper(std::string_view value)
{
    std::string out(value);
    std::ranges::transform(out, out.begin(), [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

/// Matches either an unqualified field name ("ingestTime") against any source, or a fully qualified
/// name ("source$ingestTime") exactly, mirroring Schema::getFieldByName semantics. NES uppercases all
/// identifiers, so the comparison is case-insensitive.
bool fieldMatches(std::string_view fieldName, std::string_view wanted)
{
    const std::string field = toUpper(fieldName);
    const std::string want = toUpper(wanted);
    if (field == want)
    {
        return true;
    }
    const std::string suffix = std::string(Schema::ATTRIBUTE_NAME_SEPARATOR) + want;
    return field.size() > suffix.size() && field.ends_with(suffix);
}
}

LatencySink::LatencySink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor)
    : Sink(std::move(backpressureController))
    , schema(sinkDescriptor.getSchema())
    , tupleSize(schema->getSizeOfSchemaInBytes())
    , ingestFieldName(sinkDescriptor.getFromConfig(ConfigParametersLatency::INGEST_FIELD))
    , ingestFieldOffset(std::numeric_limits<std::uint64_t>::max())
    , logPath(sinkDescriptor.getFromConfig(ConfigParametersLatency::LOG_PATH))
{
    /// Locate the ingestion-time field and its byte offset within a tuple. The null-handling byte
    /// (if the field is nullable) precedes the value and is skipped.
    std::uint64_t offset = 0;
    for (const auto& field : *schema)
    {
        if (fieldMatches(field.name, ingestFieldName))
        {
            if (!field.dataType.isType(DataType::Type::UINT64))
            {
                throw CannotOpenSink(
                    "LatencySink: ingest field '{}' must be UINT64, but got {}", field.name, field.dataType);
            }
            ingestFieldOffset = offset + (field.dataType.nullable ? 1 : 0);
            break;
        }
        offset += field.dataType.getSizeInBytesWithNull();
    }
}

void LatencySink::start(PipelineExecutionContext&)
{
    NES_DEBUG("Setting up Latency sink: {} (ingest field '{}', log '{}')", *this, ingestFieldName, logPath);
    if (ingestFieldOffset == std::numeric_limits<std::uint64_t>::max())
    {
        throw CannotOpenSink(
            "LatencySink: sink schema has no UINT64 field named '{}'; project CURRENT_TIME() AS {} into the sink.",
            ingestFieldName,
            ingestFieldName);
    }

    logStream.open(logPath, std::ofstream::out | std::ofstream::trunc);
    if (!logStream.is_open())
    {
        throw CannotOpenSink("LatencySink: could not open latency log: logPath={}", logPath);
    }
    logStream << "seq_number,chunk_number,tuple_index,ingest_ts_us,recv_ts_us,latency_us,creation_ts_ms\n";
}

void LatencySink::stop(PipelineExecutionContext&)
{
    if (logStream.is_open())
    {
        logStream.flush();
        logStream.close();
    }
    NES_INFO("Latency Sink completed. Records seen: {}", recordsSeen);
}

void LatencySink::execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext&)
{
    PRECONDITION(inputTupleBuffer, "Invalid input buffer in LatencySink.");
    const std::uint64_t numberOfTuples = inputTupleBuffer.getNumberOfTuples();
    if (numberOfTuples == 0)
    {
        return;
    }

    const auto* base = inputTupleBuffer.getAvailableMemoryArea<const std::byte>().data();
    const auto creationTsMs = static_cast<std::int64_t>(inputTupleBuffer.getCreationTimestampInMS().getRawValue());
    const auto sequenceNumber = inputTupleBuffer.getSequenceNumber().getRawValue();
    const auto chunkNumber = inputTupleBuffer.getChunkNumber().getRawValue();

    const std::scoped_lock lock(writeMutex);
    for (std::uint64_t tuple = 0; tuple < numberOfTuples; ++tuple)
    {
        const std::byte* record = base + (tuple * tupleSize);
        std::uint64_t ingestUs = 0;
        std::memcpy(&ingestUs, record + ingestFieldOffset, sizeof(std::uint64_t));

        const std::uint64_t recvUs = nowMicros();
        const std::int64_t latencyUs = static_cast<std::int64_t>(recvUs) - static_cast<std::int64_t>(ingestUs);

        logStream << sequenceNumber << ',' << chunkNumber << ',' << tuple << ',' << ingestUs << ',' << recvUs << ',' << latencyUs
                  << ',' << creationTsMs << '\n';
        ++recordsSeen;
    }
    /// Deliberately NOT flushing per buffer: a per-buffer flush turns this measurement sink into a
    /// throughput bottleneck (especially on SD-backed storage), which inflates the very latency it is
    /// meant to observe. The ofstream buffers in memory and is flushed once in stop(). Records are
    /// lost only if the query is killed mid-run, which is acceptable for benchmarking.
}

DescriptorConfig::Config LatencySink::validateAndFormat(std::unordered_map<std::string, std::string> config)
{
    return DescriptorConfig::validateAndFormat<ConfigParametersLatency>(std::move(config), NAME);
}

SinkValidationRegistryReturnType RegisterLatencySinkValidation(SinkValidationRegistryArguments sinkConfig)
{
    return LatencySink::validateAndFormat(std::move(sinkConfig.config));
}

SinkRegistryReturnType RegisterLatencySink(SinkRegistryArguments sinkRegistryArguments)
{
    return std::make_unique<LatencySink>(std::move(sinkRegistryArguments.backpressureController), sinkRegistryArguments.sinkDescriptor);
}

}
