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

#include "../include/ModelZooUDFPhysicalFunction.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <Functions/PhysicalFunction.hpp>
#include <Inference.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
constexpr auto FunctionName = "SQUEEZEDET_192X624_UDF";
constexpr auto ModelEnv = "NES_SQUEEZEDET_192X624_UDF_MODEL";
constexpr auto ModelPath = "nes-systests/testdata/model/waymo/pretrained/squeezedet/squeezedet-192x624.onnx";

std::filesystem::path modelPath()
{
    return modelZooUdfModelPath(ModelEnv, ModelPath);
}

const CompiledModel& compiledModel()
{
    static const auto model = []
    {
        const auto path = modelPath();
        auto imported = importModel(path, ModelBackend::OPENVINO);
        if (!imported)
        {
            throw CannotLoadModel("{} failed to import OpenVINO model '{}': {}", FunctionName, path.string(), imported.error().message);
        }

        auto compiled = compileModel(*imported);
        if (!compiled)
        {
            throw CannotLoadModel("{} failed to compile OpenVINO model '{}': {}", FunctionName, path.string(), compiled.error().message);
        }
        return std::move(*compiled);
    }();
    return model;
}

InferenceRuntime& runtime()
{
    thread_local auto instance = []
    {
        InferenceRuntime createdRuntime;
        createdRuntime.setup(compiledModel(), 1);
        return createdRuntime;
    }();
    return instance;
}

uint64_t outputSize()
{
    return runtime().getOutputSize();
}

uint64_t infer(int8_t* inputPtr, uint64_t inputSize, int8_t* outputPtr, uint64_t outputCapacity)
{
    auto& runtimeRef = runtime();
    const auto expectedInputSize = runtimeRef.getInputSize();
    auto* runtimeInput = runtimeRef.getInputData();

    const auto bytesToCopy = std::min<size_t>(static_cast<size_t>(inputSize), expectedInputSize);
    if (bytesToCopy > 0)
    {
        /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) NES var-sized data uses int8_t*.
        std::memcpy(runtimeInput, reinterpret_cast<std::byte*>(inputPtr), bytesToCopy);
    }
    if (bytesToCopy < expectedInputSize)
    {
        std::memset(runtimeInput + bytesToCopy, 0, expectedInputSize - bytesToCopy);
    }

    runtimeRef.infer();

    const auto runtimeOutputSize = runtimeRef.getOutputSize();
    if (runtimeOutputSize > outputCapacity)
    {
        throw InferenceRuntimeFailure(
            "{} output size {} B exceeds output buffer capacity {} B", FunctionName, runtimeOutputSize, outputCapacity);
    }
    if (runtimeOutputSize > 0)
    {
        /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) NES var-sized data uses int8_t*.
        std::memcpy(outputPtr, reinterpret_cast<int8_t*>(runtimeRef.getOutputData()), runtimeOutputSize);
    }
    return runtimeOutputSize;
}

using SqueezeDet192x624PhysicalFunction = ModelZooUDFPhysicalFunction<outputSize, infer>;
static_assert(PhysicalFunctionConcept<SqueezeDet192x624PhysicalFunction>);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterSQUEEZEDET_192X624_UDFPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 1, "{} function must have exactly one child function", FunctionName);
    return SqueezeDet192x624PhysicalFunction(arguments.childFunctions[0]);
}

}
