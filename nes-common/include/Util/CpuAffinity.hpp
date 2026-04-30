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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace NES
{
/// Parses CPU set strings like: "0", "0,1,3", "0-3,8,10-11".
/// Returns an ordered and de-duplicated list of logical CPU ids.
/// If maxCpuExclusive is set, all parsed ids must be smaller than this bound.
std::optional<std::vector<size_t>> parseCpuSet(std::string_view cpuSet, size_t maxCpuExclusive = 0);

/// Returns std::thread::hardware_concurrency() but never 0.
size_t getHardwareConcurrencyOrOne();

/// Pins current thread to a single logical CPU.
/// Returns false and sets errorMessage on failure.
bool pinCurrentThreadToCpu(size_t cpu, std::string* errorMessage = nullptr);

/// Pins current thread to a set of logical CPUs.
/// Returns false and sets errorMessage on failure.
bool pinCurrentThreadToCpuSet(const std::vector<size_t>& cpus, std::string* errorMessage = nullptr);

/// Reads current thread affinity and returns the allowed logical CPU ids.
/// Returns false and sets errorMessage on failure.
bool getCurrentThreadAffinity(std::vector<size_t>& cpus, std::string* errorMessage = nullptr);
}
