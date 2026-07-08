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

#include "../include/CurrentTimeLogicalFunction.hpp"

#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/DataTypeSerializationUtil.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>

namespace NES
{

CurrentTimeLogicalFunction::CurrentTimeLogicalFunction() : dataType(DataTypeProvider::provideDataType(DataType::Type::UINT64))
{
}

bool CurrentTimeLogicalFunction::operator==(const CurrentTimeLogicalFunction& rhs) const
{
    return dataType == rhs.dataType;
}

std::string CurrentTimeLogicalFunction::explain(ExplainVerbosity) const
{
    return "CURRENT_TIME()";
}

DataType CurrentTimeLogicalFunction::getDataType() const
{
    return dataType;
}

CurrentTimeLogicalFunction CurrentTimeLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction CurrentTimeLogicalFunction::withInferredDataType(const Schema&) const
{
    /// The data type is fixed (UINT64 microseconds); nothing to infer from the input schema.
    return *this;
}

std::vector<LogicalFunction> CurrentTimeLogicalFunction::getChildren() const
{
    return {};
}

CurrentTimeLogicalFunction CurrentTimeLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.empty(), "CURRENT_TIME function takes no children, but got {}", children.size());
    return *this;
}

std::string_view CurrentTimeLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<CurrentTimeLogicalFunction>::operator()(const CurrentTimeLogicalFunction& function) const
{
    return reflect(detail::ReflectedCurrentTimeLogicalFunction{.dataType = function.getDataType()});
}

CurrentTimeLogicalFunction Unreflector<CurrentTimeLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [dataType] = unreflect<detail::ReflectedCurrentTimeLogicalFunction>(reflected);
    return CurrentTimeLogicalFunction{}.withDataType(dataType);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterCURRENT_TIMELogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<CurrentTimeLogicalFunction>(arguments.reflected);
    }
    if (!arguments.children.empty())
    {
        throw CannotDeserialize("CURRENT_TIME function takes no arguments, but got {}", arguments.children.size());
    }
    return CurrentTimeLogicalFunction{};
}

}
