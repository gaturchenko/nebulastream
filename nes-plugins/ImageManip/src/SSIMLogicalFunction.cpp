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

#include "../include/SSIMLogicalFunction.hpp"

#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/DataTypeSerializationUtil.hpp> /// NOLINT(misc-include-cleaner)
#include <Util/PlanRenderer.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>
#include <SerializableVariantDescriptor.pb.h> /// NOLINT(misc-include-cleaner)

namespace NES
{

namespace
{
void validateSsimChildren(const LogicalFunction& image, const LogicalFunction& threshold)
{
    if (!image.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            image.getDataType().isType(DataType::Type::VARSIZED),
            "SSIM requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
    if (!threshold.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            threshold.getDataType().isFloat(),
            "SSIM requires second argument to be FLOAT32/FLOAT64, but got {}",
            threshold.getDataType());
    }
}
}

SSIMLogicalFunction::SSIMLogicalFunction(const LogicalFunction& image, const LogicalFunction& threshold)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), image(image), threshold(threshold)
{
    validateSsimChildren(image, threshold);
}

bool SSIMLogicalFunction::operator==(const LogicalFunctionConcept& rhs) const
{
    if (const auto* other = dynamic_cast<const SSIMLogicalFunction*>(&rhs))
    {
        return image == other->image and threshold == other->threshold;
    }
    return false;
}

std::string SSIMLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("SSIM({}, {})", image.explain(verbosity), threshold.explain(verbosity));
}

DataType SSIMLogicalFunction::getDataType() const
{
    return dataType;
}

LogicalFunction SSIMLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction SSIMLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (const auto& child : getChildren())
    {
        newChildren.push_back(child.withInferredDataType(schema));
    }
    return withChildren(newChildren);
}

std::vector<LogicalFunction> SSIMLogicalFunction::getChildren() const
{
    return {image, threshold};
}

LogicalFunction SSIMLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 2, "SSIMLogicalFunction requires exactly two children, but got {}", children.size());
    validateSsimChildren(children[0], children[1]);

    auto copy = *this;
    copy.image = children[0];
    copy.threshold = children[1];
    return copy;
}

std::string_view SSIMLogicalFunction::getType() const
{
    return NAME;
}

SerializableFunction SSIMLogicalFunction::serialize() const
{
    SerializableFunction serializedFunction;
    serializedFunction.set_function_type(NAME);
    serializedFunction.add_children()->CopyFrom(image.serialize());
    serializedFunction.add_children()->CopyFrom(threshold.serialize());
    DataTypeSerializationUtil::serializeDataType(getDataType(), serializedFunction.mutable_data_type());
    return serializedFunction;
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterSSIMLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (arguments.children.size() != 2)
    {
        throw CannotDeserialize("SSIMLogicalFunction requires exactly two children, but got {}", arguments.children.size());
    }
    return SSIMLogicalFunction(arguments.children[0], arguments.children[1]);
}

}
