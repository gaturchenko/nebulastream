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
#include <fstream>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <Configurations/Descriptor.hpp>
#include <DataTypes/Schema.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sinks/Sink.hpp>
#include <Sinks/SinkDescriptor.hpp>
#include <Util/Logger/Formatter.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

/// A measurement-grade "void" sink: it discards every record's payload (like the Void sink) but reads
/// one carried UINT64 ingestion-time field per record, stamps its own wall-clock receive time, and
/// appends the resulting per-record latency to a CSV. It is the sink half of the latency-measurement
/// pair; the ingestion field is produced upstream by `CURRENT_TIME()` (microseconds since the Unix
/// epoch) and carried through windows via `MAX(ingestTime)`.
///
/// Because it only reads 8 bytes per tuple and never serializes the (potentially megabyte-sized)
/// payload, its per-record overhead is negligible next to real processing — run once against the plain
/// Void sink to confirm throughput is unchanged, which keeps the "pure processing" baseline honest
/// while still yielding a real per-record latency distribution.
///
/// Output CSV columns:
///   seq_number, chunk_number, tuple_index, ingest_ts_us, recv_ts_us, latency_us, creation_ts_ms
/// where `latency_us = recv_ts_us - ingest_ts_us` (both wall-clock std::system_clock microseconds).
class LatencySink final : public Sink
{
public:
    static constexpr std::string_view NAME = "Latency";
    explicit LatencySink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor);
    ~LatencySink() override = default;

    LatencySink(const LatencySink&) = delete;
    LatencySink& operator=(const LatencySink&) = delete;
    LatencySink(LatencySink&&) = delete;
    LatencySink& operator=(LatencySink&&) = delete;

    void start(PipelineExecutionContext&) override;
    void stop(PipelineExecutionContext&) override;
    void execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext&) override;
    static DescriptorConfig::Config validateAndFormat(std::unordered_map<std::string, std::string> config);

protected:
    std::ostream& toString(std::ostream& os) const override { return os << "LatencySink"; }

private:
    std::shared_ptr<const Schema> schema;
    std::size_t tupleSize;
    std::string ingestFieldName;
    /// Byte offset within a tuple of the UINT64 ingestion-time value (null-handling byte skipped).
    std::uint64_t ingestFieldOffset;

    std::string logPath;
    std::ofstream logStream;
    std::mutex writeMutex;
    std::uint64_t recordsSeen = 0;
};

/// NOLINTBEGIN(cert-err58-cpp)
struct ConfigParametersLatency
{
    /// Path to the per-record latency CSV written by the sink.
    static inline const DescriptorConfig::ConfigParameter<std::string> LOG_PATH{
        "log_path",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(LOG_PATH, config); }};

    /// Name of the UINT64 field carrying the ingestion timestamp (as produced by CURRENT_TIME()).
    /// Matched unqualified (any source) or fully qualified ("source$ingestTime").
    static inline const DescriptorConfig::ConfigParameter<std::string> INGEST_FIELD{
        "ingest_field",
        std::string("ingestTime"),
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(INGEST_FIELD, config); }};

    static inline std::unordered_map<std::string, DescriptorConfig::ConfigParameterContainer> parameterMap
        = DescriptorConfig::createConfigParameterContainerMap(LOG_PATH, INGEST_FIELD);
};
/// NOLINTEND(cert-err58-cpp)

}

FMT_OSTREAM(NES::LatencySink);
