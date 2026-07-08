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

#include "../include/CurrentTimePhysicalFunction.hpp"

#include <chrono>
#include <cstdint>

#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
/// Microseconds since the Unix epoch on the wall clock (std::system_clock, same clock the Latency sink
/// stamps its receive time with). Read fresh on every tuple; invoked opaquely via nautilus::invoke so
/// the compiler cannot fold two calls into one constant.
uint64_t nowMicros()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
}

VarVal CurrentTimePhysicalFunction::execute(const Record&, ArenaRef&) const
{
    const nautilus::val<uint64_t> now = nautilus::invoke(nowMicros);
    return VarVal(now);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterCURRENT_TIMEPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(
        arguments.childFunctions.empty(), "CURRENT_TIME function takes no child functions, but got {}", arguments.childFunctions.size());
    PRECONDITION(
        arguments.inputTypes.empty(), "CURRENT_TIME function takes no input types, but got {}", arguments.inputTypes.size());
    return CurrentTimePhysicalFunction{};
}

}
