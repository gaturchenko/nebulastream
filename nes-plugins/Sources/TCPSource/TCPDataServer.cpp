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

#include <TCPDataServer.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <boost/asio.hpp> ///NOLINT(misc-include-cleaner)
#include <boost/asio/buffer.hpp>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/impl/write.hpp>
#include <boost/asio/post.hpp>
#include <boost/system/detail/error_code.hpp>

#include <ErrorHandling.hpp>

namespace NES
{
namespace
{
class SendRateLimiter
{
public:
    explicit SendRateLimiter(const std::optional<double> tuplesPerSecond)
    {
        if (tuplesPerSecond.has_value() and tuplesPerSecond.value() > 0)
        {
            tupleInterval = std::chrono::ceil<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / tuplesPerSecond.value()));
        }
    }

    void waitBeforeNextTuple()
    {
        if (not tupleInterval.has_value())
        {
            return;
        }

        if (firstTuple)
        {
            firstTuple = false;
            nextSend = std::chrono::steady_clock::now();
            return;
        }

        nextSend += *tupleInterval;
        std::this_thread::sleep_until(nextSend);
    }

private:
    bool firstTuple{true};
    std::optional<std::chrono::steady_clock::duration> tupleInterval;
    std::chrono::steady_clock::time_point nextSend;
};
}

TCPDataServer::TCPDataServer(std::vector<std::string> tuples, std::optional<double> tuplesPerSecond)
    : acceptor(io_context, tcp::endpoint(tcp::v4(), 0)), work_guard(boost::asio::make_work_guard(io_context))
{
    dataProvider = [tuples = std::move(tuples), tuplesPerSecond](tcp::socket& socket)
    {
        SendRateLimiter rateLimiter{tuplesPerSecond};
        for (const auto& tuple : tuples)
        {
            std::string data = tuple + "\n";
            rateLimiter.waitBeforeNextTuple();
            boost::asio::write(socket, boost::asio::buffer(data));
        }
    };
}

TCPDataServer::TCPDataServer(std::filesystem::path filePath, std::optional<double> tuplesPerSecond)
    : acceptor(io_context, tcp::endpoint(tcp::v4(), 0)), work_guard(boost::asio::make_work_guard(io_context))
{
    if (not std::filesystem::exists(filePath))
    {
        throw TestException("File to serve TCP data from does not exist: {}", filePath.string());
    }

    dataProvider = [filePath = std::move(filePath), tuplesPerSecond](tcp::socket& socket)
    {
        std::ifstream file(filePath);
        if (not file.is_open())
        {
            throw TestException("Failed to open file to serve TCP data from: {}", filePath.string());
        }

        SendRateLimiter rateLimiter{tuplesPerSecond};
        std::string line;
        while (std::getline(file, line))
        {
            std::string data = line + "\n";
            rateLimiter.waitBeforeNextTuple();
            boost::asio::write(socket, boost::asio::buffer(data));
        }
    };
}

void TCPDataServer::run(const std::stop_token& stopToken)
{
    const std::stop_callback stopCallback(stopToken, [this]() { stop(); });

    startAccept(stopToken);
    io_context.run();
}

void TCPDataServer::startAccept(const std::stop_token& stopToken)
{
    if (stopToken.stop_requested())
    {
        return;
    }

    auto socket = std::make_shared<tcp::socket>(io_context);

    acceptor.async_accept(
        *socket,
        [this, socket, stopToken](const boost::system::error_code& error)
        {
            if (not error and not stopToken.stop_requested())
            {
                handleConnection(socket, stopToken);
                startAccept(stopToken);
            }
        });
}

void TCPDataServer::handleConnection(const std::shared_ptr<tcp::socket>& socket, const std::stop_token& stopToken)
{
    /// Use async operations to handle the connection without blocking
    boost::asio::post(
        io_context,
        [this, socket, stopToken]()
        {
            if (stopToken.stop_requested())
            {
                return;
            }

            try
            {
                /// Serve the data to the client
                dataProvider(*socket);

                boost::system::error_code boostErrorCode;
                socket->shutdown(tcp::socket::shutdown_send, boostErrorCode);
                boostErrorCode.clear();
                socket->close(boostErrorCode);
            }
            catch (const std::exception& exception)
            {
                boost::system::error_code boostErrorCode;
                socket->close(boostErrorCode);
                if (boostErrorCode)
                {
                    NES_WARNING("Failed to close socket of TCPDataServer: {}", boostErrorCode.message());
                }
                NES_DEBUG("Stopped TCPDataServer connection after exception: {}", exception.what());
            }
        });
}

void TCPDataServer::stop()
{
    boost::system::error_code boostErrorCode;
    if (const auto errorCode = acceptor.cancel(boostErrorCode); errorCode.failed())
    {
        throw TestException("Failed to cancel acceptor: {}", errorCode.message());
    }
    work_guard.reset();
    io_context.stop();
}

}
