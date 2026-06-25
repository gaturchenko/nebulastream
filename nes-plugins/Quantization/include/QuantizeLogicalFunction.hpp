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

/// Quantizes a FLOAT32 value to a lower precision format and de-quantizes it back to FLOAT32.
/// Signature: QUANTIZE(value: FLOAT32, mode: VARSIZED) -> FLOAT32
class QuantizeLogicalFunction final
{
public:
    static constexpr std::string_view NAME = "QUANTIZE";

    QuantizeLogicalFunction(const LogicalFunction& value, const LogicalFunction& mode);

    [[nodiscard]] bool operator==(const QuantizeLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] QuantizeLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] QuantizeLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    DataType dataType;
    LogicalFunction value;
    LogicalFunction mode;

    friend Reflector<QuantizeLogicalFunction>;
};

static_assert(LogicalFunctionConcept<QuantizeLogicalFunction>);

template <>
struct Reflector<QuantizeLogicalFunction>
{
    Reflected operator()(const QuantizeLogicalFunction& function) const;
};

template <>
struct Unreflector<QuantizeLogicalFunction>
{
    QuantizeLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedQuantizeLogicalFunction
{
    std::optional<LogicalFunction> value;
    std::optional<LogicalFunction> mode;
};
}

FMT_OSTREAM(NES::QuantizeLogicalFunction);
