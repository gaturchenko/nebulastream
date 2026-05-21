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

/// Logical image color conversion function.
/// Signature: CONVERT_COLOR_SCHEME(image: VARSIZED, conversionMode: VARSIZED) -> VARSIZED
/// Supported conversionMode values are case-insensitive:
/// RGB_TO_BRG, BRG_TO_RGB, BRG_TO_GRAYSCALE, GRAYSCALE_TO_BRG, RGB_TO_GRAYSCALE, GRAYSCALE_TO_RGB.
class ConvertColorSchemeLogicalFunction final
{
public:
    static constexpr std::string_view NAME = "CONVERT_COLOR_SCHEME";

    ConvertColorSchemeLogicalFunction(const LogicalFunction& image, const LogicalFunction& conversionMode);

    [[nodiscard]] bool operator==(const ConvertColorSchemeLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] ConvertColorSchemeLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] ConvertColorSchemeLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    DataType dataType;
    LogicalFunction image;
    LogicalFunction conversionMode;

    friend Reflector<ConvertColorSchemeLogicalFunction>;
};

static_assert(LogicalFunctionConcept<ConvertColorSchemeLogicalFunction>);

template <>
struct Reflector<ConvertColorSchemeLogicalFunction>
{
    Reflected operator()(const ConvertColorSchemeLogicalFunction& function) const;
};

template <>
struct Unreflector<ConvertColorSchemeLogicalFunction>
{
    ConvertColorSchemeLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedConvertColorSchemeLogicalFunction
{
    std::optional<LogicalFunction> image;
    std::optional<LogicalFunction> conversionMode;
};
}

FMT_OSTREAM(NES::ConvertColorSchemeLogicalFunction);
