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

#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Arena.hpp>

namespace NES
{

/// Physical image color conversion function.
/// Signature: ConvertColorScheme(image: VARSIZED, conversionMode: VARSIZED) -> VARSIZED
/// Supported conversionMode values (case-insensitive, must otherwise match enum token spelling):
/// RGB_TO_BRG, BRG_TO_RGB, BRG_TO_GRAYSCALE, GRAYSCALE_TO_BRG, RGB_TO_GRAYSCALE, GRAYSCALE_TO_RGB.
class ConvertColorSchemePhysicalFunction final : public PhysicalFunctionConcept
{
public:
    ConvertColorSchemePhysicalFunction(PhysicalFunction imagePhysicalFunction, PhysicalFunction conversionModePhysicalFunction);
    [[nodiscard]] VarVal execute(const Record& record, ArenaRef& arena) const override;

private:
    PhysicalFunction imagePhysicalFunction;
    PhysicalFunction conversionModePhysicalFunction;
};

}
