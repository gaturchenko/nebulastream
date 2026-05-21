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

#include <memory>

#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Arena.hpp>

namespace NES
{

struct SSIMFunctionState;

/// Stateful SSIM-based image filter.
/// Keeps the previous image and emits either the previous image when SSIM >= threshold or the current image otherwise.
class SSIMPhysicalFunction final
{
public:
    SSIMPhysicalFunction(PhysicalFunction imagePhysicalFunction, PhysicalFunction thresholdPhysicalFunction);

    [[nodiscard]] VarVal execute(const Record& record, ArenaRef& arena) const;

private:
    PhysicalFunction imagePhysicalFunction;
    PhysicalFunction thresholdPhysicalFunction;
    mutable std::shared_ptr<SSIMFunctionState> state;
};

static_assert(PhysicalFunctionConcept<SSIMPhysicalFunction>);

}
