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

#include "../include/NanodetUDFPhysicalFunction.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <Functions/PhysicalFunction.hpp>
#include <Inference.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <nautilus/std/cstring.h>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>
#include <val_arith.hpp>

#ifndef NES_NANODET_UDF_SOURCE_DIR
#define NES_NANODET_UDF_SOURCE_DIR ""
#endif

#ifndef NES_NANODET_UDF_BINARY_DIR
#define NES_NANODET_UDF_BINARY_DIR ""
#endif

namespace NES
{

namespace
{
constexpr auto NanodetModelEnv = "NES_NANODET_UDF_MODEL";
constexpr auto NanodetModelRelativePath = "model/osu/pretrained/nanodet/nanodet.onnx";
constexpr auto NanodetModelRepoPath = "nes-systests/testdata/model/osu/pretrained/nanodet/nanodet.onnx";
constexpr auto NanodetModelReleaseBuildPath
    = "cmake-build-release/nes-systests/testdata/model/osu/pretrained/nanodet/nanodet.onnx";
constexpr auto NanodetModelDebugBuildPath
    = "cmake-build-debug/nes-systests/testdata/model/osu/pretrained/nanodet/nanodet.onnx";
constexpr auto NanodetModelRelWithDebInfoBuildPath
    = "cmake-build-relwithdebinfo/nes-systests/testdata/model/osu/pretrained/nanodet/nanodet.onnx";

std::filesystem::path nanodetModelPath()
{
    const auto toAbsolutePath = [](const std::filesystem::path& path)
    {
        std::error_code error;
        const auto absolutePath = std::filesystem::absolute(path, error);
        if (error)
        {
            return path;
        }

        const auto canonicalPath = std::filesystem::weakly_canonical(absolutePath, error);
        return error ? absolutePath : canonicalPath;
    };

    if (const auto* path = std::getenv(NanodetModelEnv); path != nullptr && std::strlen(path) > 0)
    {
        return toAbsolutePath(path);
    }

    const auto sourceDir = std::filesystem::path{NES_NANODET_UDF_SOURCE_DIR};
    const auto binaryDir = std::filesystem::path{NES_NANODET_UDF_BINARY_DIR};
    const std::array<std::filesystem::path, 8> candidates{
        binaryDir / NanodetModelRepoPath,
        sourceDir / NanodetModelRepoPath,
        sourceDir / NanodetModelReleaseBuildPath,
        sourceDir / NanodetModelDebugBuildPath,
        sourceDir / NanodetModelRelWithDebInfoBuildPath,
        NanodetModelRelativePath,
        NanodetModelRepoPath,
        NanodetModelRelWithDebInfoBuildPath};
    for (const auto& candidate : candidates)
    {
        if (std::filesystem::exists(candidate))
        {
            return toAbsolutePath(candidate);
        }
    }
    return toAbsolutePath(NanodetModelRepoPath);
}

const CompiledModel& nanodetCompiledModel()
{
    static const auto model = []
    {
        const auto modelPath = nanodetModelPath();
        auto imported = importModel(modelPath, ModelBackend::OPENVINO);
        if (!imported)
        {
            throw CannotLoadModel(
                "NANODET_UDF failed to import OpenVINO model '{}': {}", modelPath.string(), imported.error().message);
        }

        auto compiled = compileModel(*imported);
        if (!compiled)
        {
            throw CannotLoadModel(
                "NANODET_UDF failed to compile OpenVINO model '{}': {}", modelPath.string(), compiled.error().message);
        }
        return std::move(*compiled);
    }();
    return model;
}

InferenceRuntime& nanodetRuntime()
{
    thread_local auto runtime = []
    {
        InferenceRuntime createdRuntime;
        createdRuntime.setup(nanodetCompiledModel(), 1);
        return createdRuntime;
    }();
    return runtime;
}

uint64_t nanodetUdfOutputSize()
{
    return nanodetRuntime().getOutputSize();
}

uint64_t nanodetUdfInfer(int8_t* inputPtr, uint64_t inputSize, int8_t* outputPtr, uint64_t outputCapacity)
{
    auto& runtime = nanodetRuntime();
    const auto expectedInputSize = runtime.getInputSize();
    auto* runtimeInput = runtime.getInputData();

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

    runtime.infer();

    const auto outputSize = runtime.getOutputSize();
    if (outputSize > outputCapacity)
    {
        throw InferenceRuntimeFailure(
            "NANODET_UDF output size {} B exceeds output buffer capacity {} B", outputSize, outputCapacity);
    }
    if (outputSize > 0)
    {
        /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) NES var-sized data uses int8_t*.
        std::memcpy(outputPtr, reinterpret_cast<int8_t*>(runtime.getOutputData()), outputSize);
    }
    return outputSize;
}
}

NanodetUDFPhysicalFunction::NanodetUDFPhysicalFunction(PhysicalFunction childPhysicalFunction)
    : childPhysicalFunction(std::move(childPhysicalFunction))
{
}

VarVal NanodetUDFPhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto input = childPhysicalFunction.execute(record, arena);
    if (input.isNullable() && input.isNull())
    {
        auto nullOutput = arena.allocateVariableSizedData(nautilus::val<uint64_t>(0));
        return VarVal{nullOutput, true, true};
    }

    const auto inputValue = input.getRawValueAs<VariableSizedData>();
    const auto outputCapacity = nautilus::invoke(nanodetUdfOutputSize);
    auto output = arena.allocateVariableSizedData(outputCapacity);
    const auto actualOutputSize
        = nautilus::invoke(nanodetUdfInfer, inputValue.getContent(), inputValue.getSize(), output.getContent(), outputCapacity);
    return VarVal{VariableSizedData(output.getContent(), actualOutputSize), input.isNullable(), false};
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterNANODET_UDFPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 1, "NANODET_UDF function must have exactly one child function");
    return NanodetUDFPhysicalFunction(arguments.childFunctions[0]);
}

}
