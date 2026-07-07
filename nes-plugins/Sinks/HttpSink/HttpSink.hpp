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
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <Configurations/Descriptor.hpp>
#include <DataTypes/Schema.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sinks/Sink.hpp>
#include <Sinks/SinkDescriptor.hpp>
#include <Util/Logger/Formatter.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

/// A sink that ships the VARSIZED fields of each incoming record to an external HTTP server
/// (e.g. TorchServe) via a raw-socket HTTP/1.1 POST, and records per-record timing for latency
/// and throughput measurement.
///
/// The sink is registered with NATIVE output format (it does NOT declare `output_format`), so the
/// query engine forwards raw tuple buffers with the variable-sized child buffers intact instead of
/// running the CSV OutputFormatter. Each VARSIZED field (a float array) is sent as raw little-endian
/// bytes; every request carries all VARSIZED fields of one record, matching the post-join batch of N
/// streams.
///
/// Request body framing (application/octet-stream):
///   uint32 numArrays (LE)
///   repeated numArrays times:
///     uint32 byteLength (LE)
///     byteLength raw bytes (the field content, i.e. IEEE-754 float32 little-endian)
///
/// Requests are issued synchronously over a single persistent keep-alive connection guarded by a
/// mutex; the round-trip therefore applies natural backpressure upstream. This serializes requests,
/// which is the honest baseline against a single-worker TorchServe. For concurrent serving, run
/// multiple sink instances / connections.
class HttpSink final : public Sink
{
public:
    static constexpr std::string_view NAME = "Http";
    explicit HttpSink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor);
    ~HttpSink() override = default;

    HttpSink(const HttpSink&) = delete;
    HttpSink& operator=(const HttpSink&) = delete;
    HttpSink(HttpSink&&) = delete;
    HttpSink& operator=(HttpSink&&) = delete;

    void start(PipelineExecutionContext&) override;
    void stop(PipelineExecutionContext&) override;
    void execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext&) override;
    static DescriptorConfig::Config validateAndFormat(std::unordered_map<std::string, std::string> config);

protected:
    std::ostream& toString(std::ostream& os) const override { return os << "HttpSink"; }

private:
    /// (Re)establishes the persistent TCP connection to the HTTP server. Throws CannotOpenSink on failure.
    void connectSocket();
    /// Closes the persistent connection if open.
    void closeSocket() noexcept;
    /// Sends one POST with the given body over the persistent connection and drains the response.
    /// Returns the HTTP status code, or -1 if the connection failed (caller may reconnect + retry).
    /// Sets @p responseBytes to the response Content-Length on success.
    [[nodiscard]] int postRecord(const std::string& body, std::size_t& responseBytes);

    std::shared_ptr<const Schema> schema;
    std::size_t tupleSize;
    /// Byte offset within a tuple of each VARSIZED field's inline VariableSizedAccess (null byte skipped).
    std::vector<std::uint64_t> varSizedOffsets;

    std::string host;
    std::uint16_t port;
    std::string endpointPath;
    std::string logPath;

    int socketFd = -1;
    std::mutex sendMutex;
    std::ofstream logStream;
    std::uint64_t recordsSent = 0;
};

/// NOLINTBEGIN(cert-err58-cpp)
struct ConfigParametersHttp
{
    static inline const DescriptorConfig::ConfigParameter<std::string> HOST{
        "host",
        std::string("127.0.0.1"),
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(HOST, config); }};

    static inline const DescriptorConfig::ConfigParameter<std::size_t> PORT{
        "port",
        std::size_t{8080},
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(PORT, config); }};

    /// Model name used to build the endpoint path "/predictions/<model>".
    static inline const DescriptorConfig::ConfigParameter<std::string> MODEL{
        "model",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(MODEL, config); }};

    /// Path to the per-record timing CSV written by the sink.
    static inline const DescriptorConfig::ConfigParameter<std::string> LOG_PATH{
        "log_path",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(LOG_PATH, config); }};

    static inline std::unordered_map<std::string, DescriptorConfig::ConfigParameterContainer> parameterMap
        = DescriptorConfig::createConfigParameterContainerMap(HOST, PORT, MODEL, LOG_PATH);
};
/// NOLINTEND(cert-err58-cpp)

}

FMT_OSTREAM(NES::HttpSink);
