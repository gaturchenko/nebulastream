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

namespace NES
{

/// Logical function for hardcoded OpenVINO UDF baselines.
/// Each registered SQL function accepts one VARSIZED field containing raw input tensor bytes and
/// returns one VARSIZED field containing raw output tensor bytes.
class ModelZooUDFLogicalFunction final
{
public:
    ModelZooUDFLogicalFunction(std::string functionName, const LogicalFunction& child);

    [[nodiscard]] bool operator==(const ModelZooUDFLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] ModelZooUDFLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] ModelZooUDFLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    std::string functionName;
    DataType dataType;
    LogicalFunction child;

    friend Reflector<ModelZooUDFLogicalFunction>;
};

static_assert(LogicalFunctionConcept<ModelZooUDFLogicalFunction>);

template <>
struct Reflector<ModelZooUDFLogicalFunction>
{
    Reflected operator()(const ModelZooUDFLogicalFunction& function) const;
};

template <>
struct Unreflector<ModelZooUDFLogicalFunction>
{
    ModelZooUDFLogicalFunction operator()(const Reflected& reflected) const;
};
}

namespace NES::detail
{
struct ReflectedModelZooUDFLogicalFunction
{
    std::optional<std::string> functionName;
    std::optional<LogicalFunction> child;
};
}

FMT_OSTREAM(NES::ModelZooUDFLogicalFunction);
