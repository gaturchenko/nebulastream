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

#include "../include/QuantizePhysicalFunction.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float4_e2m1.hpp>
#include <openvino/core/type/float8_e4m3.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
bool equalsIgnoreCase(std::string_view lhs, std::string_view rhs)
{
    return lhs.size() == rhs.size()
        and std::ranges::equal(
               lhs,
               rhs,
               [](const char left, const char right)
               { return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right)); });
}

float quantizeUint8(const float value)
{
    if (std::isnan(value))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const auto clamped = std::clamp(value, 0.0F, 255.0F);
    return static_cast<float>(static_cast<uint8_t>(std::nearbyint(clamped)));
}

float quantizeFloat32(float value, const int8_t* modeData, uint64_t modeSize)
{
    if (modeData == nullptr or modeSize == 0U)
    {
        throw CannotFormatMalformedStringValue("QUANTIZE mode must not be empty");
    }

    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) mode bytes are parser-owned ASCII string data.
    const std::string_view mode(reinterpret_cast<const char*>(modeData), modeSize);
    if (equalsIgnoreCase(mode, "bfloat16") or equalsIgnoreCase(mode, "bf16"))
    {
        return static_cast<float>(ov::bfloat16(value));
    }
    if (equalsIgnoreCase(mode, "float8") or equalsIgnoreCase(mode, "float8_e4m3") or equalsIgnoreCase(mode, "f8"))
    {
        return static_cast<float>(ov::float8_e4m3(value));
    }
    if (equalsIgnoreCase(mode, "float4") or equalsIgnoreCase(mode, "float4_e2m1") or equalsIgnoreCase(mode, "f4"))
    {
        return static_cast<float>(ov::float4_e2m1(value));
    }
    if (equalsIgnoreCase(mode, "uint8") or equalsIgnoreCase(mode, "u8"))
    {
        return quantizeUint8(value);
    }
    throw CannotFormatMalformedStringValue("Unsupported QUANTIZE mode '{}'", mode);
}
}

QuantizePhysicalFunction::QuantizePhysicalFunction(PhysicalFunction valuePhysicalFunction, PhysicalFunction modePhysicalFunction)
    : valuePhysicalFunction(std::move(valuePhysicalFunction)), modePhysicalFunction(std::move(modePhysicalFunction))
{
}

VarVal QuantizePhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto value
        = valuePhysicalFunction.execute(record, arena).castToType(DataType::Type::FLOAT32).getRawValueAs<nautilus::val<float>>();
    const auto mode = modePhysicalFunction.execute(record, arena).getRawValueAs<VariableSizedData>();
    const auto quantizedValue = nautilus::invoke(quantizeFloat32, value, mode.getContent(), mode.getSize());
    return VarVal{quantizedValue};
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterQUANTIZEPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 2, "QUANTIZE function must have exactly two child functions");
    PRECONDITION(arguments.inputTypes.size() == 2, "QUANTIZE function expects exactly two input type descriptors");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::FLOAT32),
        "QUANTIZE first argument must be FLOAT32, but got {}",
        arguments.inputTypes[0]);
    PRECONDITION(
        arguments.inputTypes[1].isType(DataType::Type::VARSIZED),
        "QUANTIZE second argument must be VARSIZED, but got {}",
        arguments.inputTypes[1]);

    return QuantizePhysicalFunction(arguments.childFunctions[0], arguments.childFunctions[1]);
}

}
