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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Util/Logger/Formatter.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <SerializableVariantDescriptor.pb.h>

namespace NES
{

/// Logical spectral subtraction audio denoising function for raw little-endian float32 PCM in VARSIZED data.
/// Signature: SPECTRAL_SUBTRACTION(audio: VARSIZED, noiseDbfs: FLOAT32/FLOAT64) -> VARSIZED
class SpectralSubtractionLogicalFunction final
{
public:
    static constexpr std::string_view NAME = "SPECTRAL_SUBTRACTION";

    SpectralSubtractionLogicalFunction(const LogicalFunction& audio, const LogicalFunction& noiseDbfs);

    [[nodiscard]] bool operator==(const SpectralSubtractionLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] SpectralSubtractionLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] SpectralSubtractionLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    DataType dataType;
    LogicalFunction audio;
    LogicalFunction noiseDbfs;

    friend Reflector<SpectralSubtractionLogicalFunction>;
};

static_assert(LogicalFunctionConcept<SpectralSubtractionLogicalFunction>);

template <>
struct Reflector<SpectralSubtractionLogicalFunction>
{
    Reflected operator()(const SpectralSubtractionLogicalFunction& function) const;
};

template <>
struct Unreflector<SpectralSubtractionLogicalFunction>
{
    SpectralSubtractionLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedSpectralSubtractionLogicalFunction
{
    std::optional<LogicalFunction> audio;
    std::optional<LogicalFunction> noiseDbfs;
};
}

FMT_OSTREAM(NES::SpectralSubtractionLogicalFunction);
