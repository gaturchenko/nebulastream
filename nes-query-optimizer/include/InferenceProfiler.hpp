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

#include <expected>

#include <Identifiers/Identifiers.hpp>
#include <ErrorHandling.hpp>
#include <ModelProfile.hpp>

namespace NES
{

/// Submits model profiling jobs to workers. The optimizer uses the returned profiles to choose between physical inference
/// alternatives for the worker an inference operator is placed on.
class InferenceProfiler
{
public:
    virtual ~InferenceProfiler() = default;
    [[nodiscard]] virtual std::expected<ModelProfile, Exception> profile(const Host& worker, const ModelProfilingRequest& request) = 0;
};

}
