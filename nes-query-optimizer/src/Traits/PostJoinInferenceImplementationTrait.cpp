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

#include <Traits/PostJoinInferenceImplementationTrait.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <typeinfo>

#include <Traits/Trait.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include <TraitRegisty.hpp>

namespace NES
{
/// Required for plugin registration, no implementation necessary
TraitRegistryReturnType TraitGeneratedRegistrar::RegisterPostJoinInferenceImplementationTrait(TraitRegistryArguments arguments)
{
    return unreflect<PostJoinInferenceImplementationTrait>(arguments.reflected);
}

PostJoinInferenceImplementationTrait::PostJoinInferenceImplementationTrait(const PostJoinInferenceImplementation implementation)
    : implementation(implementation)
{
}

const std::type_info& PostJoinInferenceImplementationTrait::getType() const
{
    return typeid(PostJoinInferenceImplementationTrait);
}

bool PostJoinInferenceImplementationTrait::operator==(const PostJoinInferenceImplementationTrait& other) const
{
    return implementation == other.implementation;
}

size_t PostJoinInferenceImplementationTrait::hash() const
{
    return magic_enum::enum_integer(implementation);
}

std::string PostJoinInferenceImplementationTrait::explain(ExplainVerbosity) const
{
    return fmt::format("PostJoinInferenceImplementationTrait: {}", magic_enum::enum_name(implementation));
}

std::string_view PostJoinInferenceImplementationTrait::getName() const
{
    return NAME;
}

Reflected Reflector<PostJoinInferenceImplementationTrait>::operator()(const PostJoinInferenceImplementationTrait& trait) const
{
    return reflect(detail::ReflectedPostJoinInferenceImplementationTrait{trait.implementation});
}

PostJoinInferenceImplementationTrait Unreflector<PostJoinInferenceImplementationTrait>::operator()(const Reflected& reflected) const
{
    auto [implementation] = unreflect<detail::ReflectedPostJoinInferenceImplementationTrait>(reflected);
    return PostJoinInferenceImplementationTrait{implementation};
}

}
