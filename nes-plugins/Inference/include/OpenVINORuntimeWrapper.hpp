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
#include <span>
#include <string>
#include <vector>
#include <DataTypes/DataType.hpp>
#include <openvino/openvino.hpp>

namespace NES
{
struct OpenVINOExecutionConfig
{
    uint64_t inferenceNumThreads = 1;
    uint64_t numStreams = 1;
    bool enableCpuPinning = false;
};

class OpenVINORuntimeWrapper
{
public:
    void setup(
        const std::string& modelXml,
        std::span<const std::byte> modelBin,
        const ov::element::Type& inputElementType,
        const std::vector<size_t>& inputShape,
        OpenVINOExecutionConfig executionConfig);

    void execute(const void* inputData, void* outputData);

private:
    ov::InferRequest inferRequest;
    ov::element::Type inputElementType;
    ov::Shape inputShape;
    ov::element::Type outputElementType;
    ov::Shape outputShape;
    size_t outputTensorSize = 0;
    OpenVINOExecutionConfig executionConfig;
};

}
