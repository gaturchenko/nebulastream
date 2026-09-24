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

#include <Traits/PredictionCacheScopeTrait.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <typeinfo>

#include <Traits/Trait.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include <InferenceConfiguration.hpp>
#include <TraitRegisty.hpp>

namespace NES
{
/// Required for plugin registration, no implementation necessary
TraitRegistryReturnType TraitGeneratedRegistrar::RegisterPredictionCacheScopeTrait(TraitRegistryArguments arguments)
{
    return unreflect<PredictionCacheScopeTrait>(arguments.reflected);
}

PredictionCacheScopeTrait::PredictionCacheScopeTrait(const PredictionCacheScope scope) : scope(scope)
{
}

const std::type_info& PredictionCacheScopeTrait::getType() const
{
    return typeid(PredictionCacheScopeTrait);
}

bool PredictionCacheScopeTrait::operator==(const PredictionCacheScopeTrait& other) const
{
    return scope == other.scope;
}

size_t PredictionCacheScopeTrait::hash() const
{
    return magic_enum::enum_integer(scope);
}

std::string PredictionCacheScopeTrait::explain(ExplainVerbosity) const
{
    return fmt::format("PredictionCacheScopeTrait: {}", magic_enum::enum_name(scope));
}

std::string_view PredictionCacheScopeTrait::getName() const
{
    return NAME;
}

Reflected Reflector<PredictionCacheScopeTrait>::operator()(const PredictionCacheScopeTrait& trait) const
{
    return reflect(detail::ReflectedPredictionCacheScopeTrait{trait.scope});
}

PredictionCacheScopeTrait Unreflector<PredictionCacheScopeTrait>::operator()(const Reflected& reflected) const
{
    auto [scope] = unreflect<detail::ReflectedPredictionCacheScopeTrait>(reflected);
    return PredictionCacheScopeTrait{scope};
}

}
