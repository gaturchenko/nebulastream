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

#include <Util/CpuAffinity.hpp>

#include <algorithm>
#include <thread>
#include <vector>
#include <Util/Strings.hpp>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <cstring>
#endif

namespace NES
{
namespace
{
std::optional<size_t> parseCpuId(const std::string_view token)
{
    return from_chars<size_t>(trimWhiteSpaces(token));
}
}

std::optional<std::vector<size_t>> parseCpuSet(const std::string_view cpuSet, const size_t maxCpuExclusive)
{
    const auto trimmed = trimWhiteSpaces(cpuSet);
    if (trimmed.empty())
    {
        return std::vector<size_t>{};
    }

    std::vector<size_t> result;
    for (const auto tokenRaw : splitOnMultipleDelimiters(trimmed, {','}))
    {
        const auto token = trimWhiteSpaces(tokenRaw);
        if (token.empty())
        {
            return std::nullopt;
        }

        const auto dashPos = token.find('-');
        if (dashPos == std::string_view::npos)
        {
            const auto parsed = parseCpuId(token);
            if (!parsed.has_value())
            {
                return std::nullopt;
            }
            if (maxCpuExclusive > 0 && *parsed >= maxCpuExclusive)
            {
                return std::nullopt;
            }
            result.push_back(*parsed);
            continue;
        }

        const auto lhs = parseCpuId(token.substr(0, dashPos));
        const auto rhs = parseCpuId(token.substr(dashPos + 1));
        if (!lhs.has_value() || !rhs.has_value() || *lhs > *rhs)
        {
            return std::nullopt;
        }
        if (maxCpuExclusive > 0 && *rhs >= maxCpuExclusive)
        {
            return std::nullopt;
        }

        for (size_t cpu = *lhs; cpu <= *rhs; ++cpu)
        {
            result.push_back(cpu);
        }
    }

    std::ranges::sort(result);
    result.erase(std::ranges::unique(result).begin(), result.end());
    return result;
}

size_t getHardwareConcurrencyOrOne()
{
    const auto hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
}

bool pinCurrentThreadToCpuSet(const std::vector<size_t>& cpus, std::string* errorMessage)
{
#ifdef __linux__
    if (cpus.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "CPU set must not be empty";
        }
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (const auto cpu : cpus)
    {
        if (cpu >= CPU_SETSIZE)
        {
            if (errorMessage)
            {
                *errorMessage = "CPU id exceeds CPU_SETSIZE";
            }
            return false;
        }
        CPU_SET(static_cast<int>(cpu), &set);
    }

    const auto rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    if (rc != 0)
    {
        if (errorMessage)
        {
            *errorMessage = std::strerror(rc);
        }
        return false;
    }
    return true;
#else
    static_cast<void>(cpus);
    if (errorMessage)
    {
        *errorMessage = "Thread affinity pinning is only supported on Linux";
    }
    return false;
#endif
}

bool pinCurrentThreadToCpu(const size_t cpu, std::string* errorMessage)
{
    return pinCurrentThreadToCpuSet(std::vector<size_t>{cpu}, errorMessage);
}

bool getCurrentThreadAffinity(std::vector<size_t>& cpus, std::string* errorMessage)
{
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    const auto rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    if (rc != 0)
    {
        if (errorMessage)
        {
            *errorMessage = std::strerror(rc);
        }
        return false;
    }

    cpus.clear();
    for (size_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    {
        if (CPU_ISSET(static_cast<int>(cpu), &set))
        {
            cpus.push_back(cpu);
        }
    }
    return true;
#else
    cpus.clear();
    if (errorMessage)
    {
        *errorMessage = "Reading thread affinity is only supported on Linux";
    }
    return false;
#endif
}
}
