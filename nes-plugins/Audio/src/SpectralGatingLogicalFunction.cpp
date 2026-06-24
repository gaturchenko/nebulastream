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

#include "../include/SpectralGatingLogicalFunction.hpp"

#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
void validateSpectralGatingChildren(const LogicalFunction& audio, const LogicalFunction& noiseDbfs)
{
    if (!audio.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            audio.getDataType().isType(DataType::Type::VARSIZED),
            "SPECTRAL_GATING requires first argument to be VARSIZED, but got {}",
            audio.getDataType());
    }
    if (!noiseDbfs.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            noiseDbfs.getDataType().isFloat(),
            "SPECTRAL_GATING requires second argument to be FLOAT32/FLOAT64, but got {}",
            noiseDbfs.getDataType());
    }
}
}

SpectralGatingLogicalFunction::SpectralGatingLogicalFunction(const LogicalFunction& audio, const LogicalFunction& noiseDbfs)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), audio(audio), noiseDbfs(noiseDbfs)
{
    validateSpectralGatingChildren(audio, noiseDbfs);
}

bool SpectralGatingLogicalFunction::operator==(const SpectralGatingLogicalFunction& rhs) const
{
    return audio == rhs.audio && noiseDbfs == rhs.noiseDbfs;
}

std::string SpectralGatingLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("SPECTRAL_GATING({}, {})", audio.explain(verbosity), noiseDbfs.explain(verbosity));
}

DataType SpectralGatingLogicalFunction::getDataType() const
{
    return dataType;
}

SpectralGatingLogicalFunction SpectralGatingLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction SpectralGatingLogicalFunction::withInferredDataType(const Schema& schema) const
{
    auto newChildren = getChildren() | std::views::transform([&schema](const auto& child) { return child.withInferredDataType(schema); })
        | std::ranges::to<std::vector>();
    return withChildren(newChildren);
}

std::vector<LogicalFunction> SpectralGatingLogicalFunction::getChildren() const
{
    return {audio, noiseDbfs};
}

SpectralGatingLogicalFunction SpectralGatingLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 2, "SPECTRAL_GATING function requires exactly two children, but got {}", children.size());
    validateSpectralGatingChildren(children[0], children[1]);

    auto copy = *this;
    copy.audio = children[0];
    copy.noiseDbfs = children[1];
    return copy;
}

std::string_view SpectralGatingLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<SpectralGatingLogicalFunction>::operator()(const SpectralGatingLogicalFunction& function) const
{
    return reflect(detail::ReflectedSpectralGatingLogicalFunction{.audio = function.audio, .noiseDbfs = function.noiseDbfs});
}

SpectralGatingLogicalFunction Unreflector<SpectralGatingLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [audio, noiseDbfs] = unreflect<detail::ReflectedSpectralGatingLogicalFunction>(reflected);
    if (!audio.has_value() || !noiseDbfs.has_value())
    {
        throw CannotDeserialize("SPECTRAL_GATING function is missing a child");
    }
    return SpectralGatingLogicalFunction{audio.value(), noiseDbfs.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterSPECTRAL_GATINGLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<SpectralGatingLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 2)
    {
        throw CannotDeserialize("SPECTRAL_GATING function requires exactly two children, but got {}", arguments.children.size());
    }
    return SpectralGatingLogicalFunction(arguments.children[0], arguments.children[1]);
}

}
