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

/// Physical alternatives for an inference operator that consumes k compatible payload fields of one joined record.
enum class PostJoinInferenceImplementation : uint8_t
{
    /// One model invocation with a batch of k payloads per joined record.
    BATCHED,
    /// k single-tuple model invocations per joined record.
    SEQUENTIAL
};

/// Stores the physical implementation the optimizer selected for a post-join inference operator.
struct PostJoinInferenceImplementationTrait final
{
    static constexpr std::string_view NAME = "PostJoinInferenceImplementation";
    PostJoinInferenceImplementation implementation;

    explicit PostJoinInferenceImplementationTrait(PostJoinInferenceImplementation implementation);

    [[nodiscard]] const std::type_info& getType() const;

    bool operator==(const PostJoinInferenceImplementationTrait& other) const;

    [[nodiscard]] size_t hash() const;

    [[nodiscard]] std::string explain(ExplainVerbosity) const;

    [[nodiscard]] std::string_view getName() const;

    friend Reflector<PostJoinInferenceImplementationTrait>;
};

template <>
struct Reflector<PostJoinInferenceImplementationTrait>
{
    Reflected operator()(const PostJoinInferenceImplementationTrait& trait) const;
};

template <>
struct Unreflector<PostJoinInferenceImplementationTrait>
{
    PostJoinInferenceImplementationTrait operator()(const Reflected& reflected) const;
};

static_assert(TraitConcept<PostJoinInferenceImplementationTrait>);
}

namespace NES::detail
{
struct ReflectedPostJoinInferenceImplementationTrait
{
    PostJoinInferenceImplementation implementation;
};
}
