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

#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <SerializableVariantDescriptor.pb.h>
#include <Util/Logger/Formatter.hpp>
#include <Util/PlanRenderer.hpp>

namespace NES
{

/// Logical image color conversion function.
/// Signature: ConvertColorScheme(image: VARSIZED, conversionMode: VARSIZED) -> VARSIZED
/// Supported conversionMode values (case-insensitive, must otherwise match enum token spelling):
/// RGB_TO_BRG, BRG_TO_RGB, BRG_TO_GRAYSCALE, GRAYSCALE_TO_BRG, RGB_TO_GRAYSCALE, GRAYSCALE_TO_RGB.
class ConvertColorSchemeLogicalFunction final : public LogicalFunctionConcept
{
public:
    static constexpr std::string_view NAME = "ConvertColorScheme";

    ConvertColorSchemeLogicalFunction(const LogicalFunction& image, const LogicalFunction& conversionMode);

    [[nodiscard]] SerializableFunction serialize() const override;
    [[nodiscard]] bool operator==(const LogicalFunctionConcept& rhs) const override;

    [[nodiscard]] DataType getDataType() const override;
    [[nodiscard]] LogicalFunction withDataType(const DataType& dataType) const override;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const override;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const override;
    [[nodiscard]] LogicalFunction withChildren(const std::vector<LogicalFunction>& children) const override;

    [[nodiscard]] std::string_view getType() const override;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const override;

private:
    DataType dataType;
    LogicalFunction image;
    LogicalFunction conversionMode;
};

}

FMT_OSTREAM(NES::ConvertColorSchemeLogicalFunction);
