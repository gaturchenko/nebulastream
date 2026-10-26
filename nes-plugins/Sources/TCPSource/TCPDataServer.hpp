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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace NES
{

/// How the TCPDataServer serves its tuples: at a fixed rate (as fast as possible otherwise), after a delay once the source has
/// connected, and optionally each prefixed with the time it is sent (microseconds since the epoch, the clock of the Latency sink
/// and CURRENT_TIME()) as an extra first field. With the send time as a query's ingestion time, a latency includes everything the
/// tuple waits for after arriving; with the delay, the query can finish setting up before the first tuple arrives.
struct TCPDataServerPacing
{
    std::optional<double> tuplesPerSecond;
    std::chrono::milliseconds startDelay{0};
    bool prefixSendTime{false};
};

/// Takes either a vector of strings (representing tuples) or a filepath.
/// On calling 'run', waits for connections and asynchronously serves the tuples (from vector or filepath) once.
/// After serving the data, the TCPDataServer closes the connection (triggering an end of stream (EOS)).
class TCPDataServer
{
    using tcp = boost::asio::ip::tcp;
    using DataProvider = std::function<void(tcp::socket&)>;

public:
    explicit TCPDataServer(std::vector<std::string> tuples, TCPDataServerPacing pacing = {});
    explicit TCPDataServer(std::filesystem::path filePath, TCPDataServerPacing pacing = {});

    TCPDataServer(const TCPDataServer&) = delete;
    TCPDataServer operator=(const TCPDataServer&) = delete;
    TCPDataServer(const TCPDataServer&&) = delete;
    TCPDataServer operator=(const TCPDataServer&&) = delete;

    ~TCPDataServer() { stop(); }

    [[nodiscard]] uint16_t getPort() const { return acceptor.local_endpoint().port(); }

    void run(const std::stop_token& stopToken);

private:
    void handleConnection(const std::shared_ptr<tcp::socket>& socket, const std::stop_token& stopToken);
    void startAccept(const std::stop_token& stopToken);

    void stop();

    boost::asio::io_context io_context;
    tcp::acceptor acceptor;
    DataProvider dataProvider;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard;
};

}
