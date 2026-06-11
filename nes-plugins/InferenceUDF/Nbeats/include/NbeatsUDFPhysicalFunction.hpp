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

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <Arena.hpp>

#ifndef NES_NBEATS_UDF_SOURCE_DIR
#define NES_NBEATS_UDF_SOURCE_DIR ""
#endif

#ifndef NES_NBEATS_UDF_BINARY_DIR
#define NES_NBEATS_UDF_BINARY_DIR ""
#endif

namespace NES
{

[[nodiscard]] inline std::filesystem::path nbeatsUdfAbsolutePath(const std::filesystem::path& path)
{
    std::error_code error;
    const auto absolutePath = std::filesystem::absolute(path, error);
    if (error)
    {
        return path;
    }

    const auto canonicalPath = std::filesystem::weakly_canonical(absolutePath, error);
    return error ? absolutePath : canonicalPath;
}

[[nodiscard]] inline std::filesystem::path nbeatsUdfModelPath(
    const char* environmentVariable, const std::filesystem::path& modelRelativePath)
{
    if (const auto* path = std::getenv(environmentVariable); path != nullptr && std::strlen(path) > 0)
    {
        return nbeatsUdfAbsolutePath(path);
    }

    const auto sourceDir = std::filesystem::path{NES_NBEATS_UDF_SOURCE_DIR};
    const auto binaryDir = std::filesystem::path{NES_NBEATS_UDF_BINARY_DIR};
    const std::array<std::filesystem::path, 4> candidates{
        binaryDir / modelRelativePath,
        sourceDir / modelRelativePath,
        modelRelativePath,
        std::filesystem::path{"model/power/pretrained/nbeats"} / modelRelativePath.filename()};
    for (const auto& candidate : candidates)
    {
        if (std::filesystem::exists(candidate))
        {
            return nbeatsUdfAbsolutePath(candidate);
        }
    }
    return nbeatsUdfAbsolutePath(binaryDir / modelRelativePath);
}

template <
    uint64_t (*OutputSizeFunction)(),
    uint64_t (*InferFunction)(int8_t* inputPtr, uint64_t inputSize, int8_t* outputPtr, uint64_t outputCapacity)>
class NbeatsUDFPhysicalFunction final
{
public:
    explicit NbeatsUDFPhysicalFunction(PhysicalFunction childPhysicalFunction)
        : childPhysicalFunction(std::move(childPhysicalFunction))
    {
    }

    [[nodiscard]] VarVal execute(const Record& record, ArenaRef& arena) const
    {
        const auto input = childPhysicalFunction.execute(record, arena);
        if (input.isNullable() && input.isNull())
        {
            auto nullOutput = arena.allocateVariableSizedData(nautilus::val<uint64_t>(0));
            return VarVal{nullOutput, true, true};
        }

        const auto inputValue = input.getRawValueAs<VariableSizedData>();
        const auto outputCapacity = nautilus::invoke(OutputSizeFunction);
        auto output = arena.allocateVariableSizedData(outputCapacity);
        const auto actualOutputSize
            = nautilus::invoke(InferFunction, inputValue.getContent(), inputValue.getSize(), output.getContent(), outputCapacity);
        return VarVal{VariableSizedData(output.getContent(), actualOutputSize), input.isNullable(), false};
    }

private:
    PhysicalFunction childPhysicalFunction;
};

}
