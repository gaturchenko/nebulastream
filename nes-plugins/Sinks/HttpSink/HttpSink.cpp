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

#include <HttpSink.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cerrno>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <Configurations/Descriptor.hpp>
#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Runtime/VariableSizedAccess.hpp>
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

/// Milliseconds since the Unix epoch on the wall clock (comparable to TupleBuffer creation timestamps).
std::int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void appendLittleEndian(std::string& out, const std::uint32_t value)
{
    out.push_back(static_cast<char>(value & 0xFF));
    out.push_back(static_cast<char>((value >> 8) & 0xFF));
    out.push_back(static_cast<char>((value >> 16) & 0xFF));
    out.push_back(static_cast<char>((value >> 24) & 0xFF));
}

/// Resolves a VARSIZED field's inline reference to its content bytes in the parent buffer's child buffer.
/// This mirrors TupleBufferRef::loadAssociatedVarSizedValue but depends only on nes-memory primitives.
std::span<const std::byte> loadVarSized(const TupleBuffer& buffer, const VariableSizedAccess& access)
{
    auto childBuffer = buffer.loadChildBuffer(access.getIndex());
    const auto content = childBuffer.getAvailableMemoryArea<const std::byte>();
    return content.subspan(access.getOffset().getRawOffset(), access.getSize().getRawSize());
}

bool sendAll(const int fd, const char* data, const std::size_t length)
{
    std::size_t sent = 0;
    while (sent < length)
    {
        const ssize_t n = ::send(fd, data + sent, length - sent, MSG_NOSIGNAL);
        if (n <= 0)
        {
            if (n < 0 && errno == EINTR)
            {
                continue;
            }
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

}

HttpSink::HttpSink(BackpressureController backpressureController, const SinkDescriptor& sinkDescriptor)
    : Sink(std::move(backpressureController))
    , schema(sinkDescriptor.getSchema())
    , tupleSize(schema->getSizeOfSchemaInBytes())
    , host(sinkDescriptor.getFromConfig(ConfigParametersHttp::HOST))
    , port(static_cast<std::uint16_t>(sinkDescriptor.getFromConfig(ConfigParametersHttp::PORT)))
    , endpointPath("/predictions/" + sinkDescriptor.getFromConfig(ConfigParametersHttp::MODEL))
    , logPath(sinkDescriptor.getFromConfig(ConfigParametersHttp::LOG_PATH))
{
    /// Precompute the byte offset of every VARSIZED field's inline VariableSizedAccess within a tuple.
    /// The null-handling byte (if the field is nullable) precedes the value, so it is skipped.
    std::uint64_t offset = 0;
    for (const auto& field : *schema)
    {
        if (field.dataType.isType(DataType::Type::VARSIZED))
        {
            varSizedOffsets.push_back(offset + (field.dataType.nullable ? 1 : 0));
        }
        offset += field.dataType.getSizeInBytesWithNull();
    }
}

void HttpSink::connectSocket()
{
    if (socketFd >= 0)
    {
        return;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* resolved = nullptr;
    const std::string portStr = std::to_string(port);
    if (const int rc = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &resolved); rc != 0 || resolved == nullptr)
    {
        throw CannotOpenSink("HttpSink: could not resolve {}:{}: {}", host, port, ::gai_strerror(rc));
    }

    int fd = -1;
    for (const addrinfo* candidate = resolved; candidate != nullptr; candidate = candidate->ai_next)
    {
        fd = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
        if (fd < 0)
        {
            continue;
        }
        if (::connect(fd, candidate->ai_addr, candidate->ai_addrlen) == 0)
        {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(resolved);

    if (fd < 0)
    {
        throw CannotOpenSink("HttpSink: could not connect to {}:{}", host, port);
    }
    socketFd = fd;
}

void HttpSink::closeSocket() noexcept
{
    if (socketFd >= 0)
    {
        ::close(socketFd);
        socketFd = -1;
    }
}

int HttpSink::postRecord(const std::string& body, std::size_t& responseBytes)
{
    responseBytes = 0;

    std::string request;
    request.reserve(body.size() + 256);
    request += "POST " + endpointPath + " HTTP/1.1\r\n";
    request += "Host: " + host + ":" + std::to_string(port) + "\r\n";
    request += "Content-Type: application/octet-stream\r\n";
    request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    request += "Connection: keep-alive\r\n\r\n";

    if (!sendAll(socketFd, request.data(), request.size()) || !sendAll(socketFd, body.data(), body.size()))
    {
        return -1;
    }

    /// Read until we have the full header block.
    std::string response;
    std::array<char, 4096> buffer{};
    std::size_t headerEnd = std::string::npos;
    while (headerEnd == std::string::npos)
    {
        const ssize_t n = ::recv(socketFd, buffer.data(), buffer.size(), 0);
        if (n <= 0)
        {
            if (n < 0 && errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        response.append(buffer.data(), static_cast<std::size_t>(n));
        headerEnd = response.find("\r\n\r\n");
    }

    /// Parse the status code from the status line "HTTP/1.1 <code> <reason>".
    int status = 0;
    if (const auto firstSpace = response.find(' '); firstSpace != std::string::npos)
    {
        status = std::atoi(response.c_str() + firstSpace + 1);
    }

    /// Parse Content-Length (header names are case-insensitive) from the header block.
    std::string header = response.substr(0, headerEnd);
    std::ranges::transform(header, header.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::size_t contentLength = 0;
    if (const auto pos = header.find("content-length:"); pos != std::string::npos)
    {
        contentLength = static_cast<std::size_t>(std::strtoull(header.c_str() + pos + std::strlen("content-length:"), nullptr, 10));
    }

    /// Drain the body so the connection stays usable for the next keep-alive request.
    const std::size_t bodyStart = headerEnd + 4;
    std::size_t bodyReceived = response.size() - bodyStart;
    while (bodyReceived < contentLength)
    {
        const ssize_t n = ::recv(socketFd, buffer.data(), buffer.size(), 0);
        if (n <= 0)
        {
            if (n < 0 && errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        bodyReceived += static_cast<std::size_t>(n);
    }

    responseBytes = contentLength;
    return status;
}

void HttpSink::start(PipelineExecutionContext&)
{
    NES_DEBUG("Setting up HTTP sink: {} -> {}:{}{}", *this, host, port, endpointPath);
    if (varSizedOffsets.empty())
    {
        throw CannotOpenSink("HttpSink: schema has no VARSIZED fields to send; sink schema must project the array field(s).");
    }

    logStream.open(logPath, std::ofstream::out | std::ofstream::trunc);
    if (!logStream.is_open())
    {
        throw CannotOpenSink("HttpSink: could not open timing log: logPath={}", logPath);
    }
    logStream << "seq_number,chunk_number,tuple_index,creation_ts_ms,post_ts_ms,recv_ts_ms,num_arrays,request_bytes,response_bytes,"
                 "http_status\n";

    connectSocket();
}

void HttpSink::stop(PipelineExecutionContext&)
{
    closeSocket();
    if (logStream.is_open())
    {
        logStream.flush();
        logStream.close();
    }
    NES_INFO("HTTP Sink completed. Records sent: {}", recordsSent);
}

void HttpSink::execute(const TupleBuffer& inputTupleBuffer, PipelineExecutionContext&)
{
    PRECONDITION(inputTupleBuffer, "Invalid input buffer in HttpSink.");
    const std::uint64_t numberOfTuples = inputTupleBuffer.getNumberOfTuples();
    if (numberOfTuples == 0)
    {
        return;
    }

    const auto* base = inputTupleBuffer.getAvailableMemoryArea<const std::byte>().data();
    const std::int64_t creationTs = static_cast<std::int64_t>(inputTupleBuffer.getCreationTimestampInMS().getRawValue());
    const auto sequenceNumber = inputTupleBuffer.getSequenceNumber().getRawValue();
    const auto chunkNumber = inputTupleBuffer.getChunkNumber().getRawValue();
    const auto numArrays = static_cast<std::uint32_t>(varSizedOffsets.size());

    const std::scoped_lock lock(sendMutex);
    for (std::uint64_t tuple = 0; tuple < numberOfTuples; ++tuple)
    {
        const std::byte* record = base + (tuple * tupleSize);

        std::string body;
        appendLittleEndian(body, numArrays);
        for (const auto fieldOffset : varSizedOffsets)
        {
            VariableSizedAccess access;
            std::memcpy(&access, record + fieldOffset, sizeof(VariableSizedAccess));
            const auto content = loadVarSized(inputTupleBuffer, access);
            appendLittleEndian(body, static_cast<std::uint32_t>(content.size()));
            body.append(reinterpret_cast<const char*>(content.data()), content.size());
        }

        const std::int64_t postMs = nowMs();
        std::size_t responseBytes = 0;
        int status = postRecord(body, responseBytes);
        if (status < 0)
        {
            /// The persistent connection dropped (e.g. server closed keep-alive); reconnect and retry once.
            closeSocket();
            connectSocket();
            status = postRecord(body, responseBytes);
        }
        const std::int64_t recvMs = nowMs();

        logStream << sequenceNumber << ',' << chunkNumber << ',' << tuple << ',' << creationTs << ',' << postMs << ',' << recvMs << ','
                  << numArrays << ',' << body.size() << ',' << responseBytes << ',' << status << '\n';
        ++recordsSent;
    }
    logStream.flush();
}

DescriptorConfig::Config HttpSink::validateAndFormat(std::unordered_map<std::string, std::string> config)
{
    return DescriptorConfig::validateAndFormat<ConfigParametersHttp>(std::move(config), NAME);
}

SinkValidationRegistryReturnType RegisterHttpSinkValidation(SinkValidationRegistryArguments sinkConfig)
{
    return HttpSink::validateAndFormat(std::move(sinkConfig.config));
}

SinkRegistryReturnType RegisterHttpSink(SinkRegistryArguments sinkRegistryArguments)
{
    return std::make_unique<HttpSink>(std::move(sinkRegistryArguments.backpressureController), sinkRegistryArguments.sinkDescriptor);
}

}
