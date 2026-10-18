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

#include <atomic>
#include <condition_variable>
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
/// Requests are issued synchronously: a worker thread blocks on its round trip, which applies
/// natural backpressure upstream. How MANY may be in flight at once is the `connections` setting.
///
/// With `connections` = 1 (the default) every request is serialized behind one keep-alive socket.
/// That measures a single-lane client, which is only a fair counterpart to a single-worker server;
/// against a server with N workers it caps throughput at 1/service-time and charges the server for
/// a queue that is really ours. With `connections` = N the sink keeps a pool of N keep-alive
/// sockets, so up to N worker threads can have a request outstanding simultaneously. Set it to the
/// query's worker-thread count AND give the server that many workers, or the comparison measures
/// whichever side was left single-lane.
///
/// The pool mutex is held only while handing out and returning a socket -- never across a round
/// trip -- so the requests themselves genuinely overlap. A thread that finds the pool empty waits
/// for a peer to return one, which is what preserves the backpressure.
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
    /// Opens one keep-alive connection to the HTTP server. Throws CannotOpenSink on failure.
    [[nodiscard]] int openConnection() const;
    /// Closes a connection if open and sets the descriptor back to -1.
    static void closeConnection(int& fd) noexcept;
    /// Blocks until a pooled connection is free and returns its slot index.
    [[nodiscard]] std::size_t acquireConnection();
    /// Returns a slot to the pool and wakes one waiter.
    void releaseConnection(std::size_t slot) noexcept;
    /// Sends one POST with the given body over @p fd and drains the response.
    /// Returns the HTTP status code, or -1 if the connection failed (caller may reconnect + retry).
    /// Sets @p responseBytes to the response Content-Length on success.
    [[nodiscard]] int postRecord(int fd, const std::string& body, std::size_t& responseBytes) const;

    std::shared_ptr<const Schema> schema;
    std::size_t tupleSize;
    /// Byte offset within a tuple of each VARSIZED field's inline VariableSizedAccess (null byte skipped).
    std::vector<std::uint64_t> varSizedOffsets;

    std::string host;
    std::uint16_t port;
    std::string endpointPath;
    std::string logPath;

    /// Pool of keep-alive sockets. Only the thread holding a slot touches that slot's descriptor,
    /// so the descriptors themselves need no further synchronisation.
    std::size_t connections;
    std::vector<int> connectionFds;
    std::vector<std::size_t> freeConnections;
    std::mutex poolMutex;
    std::condition_variable connectionReleased;

    /// Separate from the pool: several threads finish round trips concurrently and all append here.
    std::mutex logMutex;
    std::ofstream logStream;
    std::atomic<std::uint64_t> recordsSent{0};
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

    /// Keep-alive sockets held open, i.e. the maximum number of requests in flight at once.
    /// 1 reproduces the original single-lane client; set it to the query's worker-thread count to
    /// measure a server that has that many workers.
    static inline const DescriptorConfig::ConfigParameter<std::size_t> CONNECTIONS{
        "connections",
        std::size_t{1},
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(CONNECTIONS, config); }};

    /// Path to the per-record timing CSV written by the sink.
    static inline const DescriptorConfig::ConfigParameter<std::string> LOG_PATH{
        "log_path",
        std::nullopt,
        [](const std::unordered_map<std::string, std::string>& config) { return DescriptorConfig::tryGet(LOG_PATH, config); }};

    static inline std::unordered_map<std::string, DescriptorConfig::ConfigParameterContainer> parameterMap
        = DescriptorConfig::createConfigParameterContainerMap(HOST, PORT, MODEL, CONNECTIONS, LOG_PATH);
};
/// NOLINTEND(cert-err58-cpp)

}

FMT_OSTREAM(NES::HttpSink);
