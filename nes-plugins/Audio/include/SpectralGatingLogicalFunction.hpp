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

/// Logical hard spectral gating function for raw little-endian float32 PCM in VARSIZED data.
/// Signature: SPECTRAL_GATING(audio: VARSIZED, noiseDbfs: FLOAT32/FLOAT64) -> VARSIZED
class SpectralGatingLogicalFunction final
{
public:
    static constexpr std::string_view NAME = "SPECTRAL_GATING";

    SpectralGatingLogicalFunction(const LogicalFunction& audio, const LogicalFunction& noiseDbfs);

    [[nodiscard]] bool operator==(const SpectralGatingLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] SpectralGatingLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] SpectralGatingLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    DataType dataType;
    LogicalFunction audio;
    LogicalFunction noiseDbfs;

    friend Reflector<SpectralGatingLogicalFunction>;
};

static_assert(LogicalFunctionConcept<SpectralGatingLogicalFunction>);

template <>
struct Reflector<SpectralGatingLogicalFunction>
{
    Reflected operator()(const SpectralGatingLogicalFunction& function) const;
};

template <>
struct Unreflector<SpectralGatingLogicalFunction>
{
    SpectralGatingLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedSpectralGatingLogicalFunction
{
    std::optional<LogicalFunction> audio;
    std::optional<LogicalFunction> noiseDbfs;
};
}

FMT_OSTREAM(NES::SpectralGatingLogicalFunction);
