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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <typeinfo>
#include <Traits/Trait.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>

namespace NES
{

/// Whether the worker threads running an inference operator share one compiled model instead of each compiling its own copy.
/// Sharing keeps one copy of the model's weights in memory at the price of a throughput deployment, where the streams of the
/// shared model divide one thread budget between them.
struct SharedCompiledModelTrait final
{
    static constexpr std::string_view NAME = "SharedCompiledModel";
    bool shareCompiledModel;

    explicit SharedCompiledModelTrait(bool shareCompiledModel);

    [[nodiscard]] const std::type_info& getType() const;

    bool operator==(const SharedCompiledModelTrait& other) const;

    [[nodiscard]] size_t hash() const;

    [[nodiscard]] std::string explain(ExplainVerbosity) const;

    [[nodiscard]] std::string_view getName() const;

    friend Reflector<SharedCompiledModelTrait>;
};

template <>
struct Reflector<SharedCompiledModelTrait>
{
    Reflected operator()(const SharedCompiledModelTrait& trait) const;
};

template <>
struct Unreflector<SharedCompiledModelTrait>
{
    SharedCompiledModelTrait operator()(const Reflected& reflected) const;
};

static_assert(TraitConcept<SharedCompiledModelTrait>);
}

namespace NES::detail
{
struct ReflectedSharedCompiledModelTrait
{
    bool shareCompiledModel;
};
}
