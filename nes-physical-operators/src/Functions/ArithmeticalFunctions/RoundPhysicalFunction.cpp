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

#include <Functions/ArithmeticalFunctions/RoundPhysicalFunction.hpp>

#include <utility>
#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <std/cmath.h>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>
#include <val.hpp>

namespace NES
{
RoundPhysicalFunction::RoundPhysicalFunction(PhysicalFunction childFunction, DataType inputType, DataType outputType)
    : childFunction(std::move(childFunction)), inputType(std::move(inputType)), outputType(std::move(outputType))
{
}

RoundPhysicalFunction::RoundPhysicalFunction(
    PhysicalFunction childFunction, PhysicalFunction decimalPlacesFunction, DataType inputType, DataType outputType)
    : childFunction(std::move(childFunction))
    , decimalPlacesFunction(std::move(decimalPlacesFunction))
    , inputType(std::move(inputType))
    , outputType(std::move(outputType))
{
}

VarVal RoundPhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto value = childFunction.execute(record, arena);
    /// If the input type is a float, we need to round the value and return the rounded value.
    /// If the input type is an integer, we only need to cast to the output type.
    if (inputType.isFloat())
    {
        const auto inputValue = value.getRawValueAs<nautilus::val<double>>();
        if (decimalPlacesFunction.has_value())
        {
            const auto decimalPlaces
                = decimalPlacesFunction->execute(record, arena).castToType(DataType::Type::FLOAT64).getRawValueAs<nautilus::val<double>>();
            const auto scale = nautilus::pow(nautilus::val<double>(10.0), decimalPlaces);
            const auto roundedValue = nautilus::round(inputValue * scale) / scale;
            return VarVal{roundedValue}.castToType(outputType.type);
        }
        const auto roundedValue = nautilus::round(inputValue);
        return VarVal{roundedValue}.castToType(outputType.type);
    }
    return value.castToType(outputType.type);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterRoundPhysicalFunction(PhysicalFunctionRegistryArguments physicalFunctionRegistryArguments)
{
    PRECONDITION(
        physicalFunctionRegistryArguments.childFunctions.size() == 1 or physicalFunctionRegistryArguments.childFunctions.size() == 2,
        "Round function must have one or two child functions");
    PRECONDITION(
        physicalFunctionRegistryArguments.inputTypes.size() == 1 or physicalFunctionRegistryArguments.inputTypes.size() == 2,
        "Round function must have one or two input types");
    if (physicalFunctionRegistryArguments.childFunctions.size() == 2)
    {
        return RoundPhysicalFunction(
            physicalFunctionRegistryArguments.childFunctions[0],
            physicalFunctionRegistryArguments.childFunctions[1],
            physicalFunctionRegistryArguments.inputTypes[0],
            physicalFunctionRegistryArguments.outputType);
    }
    return RoundPhysicalFunction(
        physicalFunctionRegistryArguments.childFunctions[0],
        physicalFunctionRegistryArguments.inputTypes[0],
        physicalFunctionRegistryArguments.outputType);
}

}
