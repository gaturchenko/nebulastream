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
#include <InferenceConfiguration.hpp>

namespace NES
{

/// The scope of an inference operator's prediction cache: one cache per worker thread, or a single cache shared by all of them.
/// A global cache exists once instead of once per thread, at the price of serializing the lookups of all worker threads, measured
/// at about 1/threads of the throughput of thread-local caches for the record sizes of real models.
struct PredictionCacheScopeTrait final
{
    static constexpr std::string_view NAME = "PredictionCacheScope";
    PredictionCacheScope scope;

    explicit PredictionCacheScopeTrait(PredictionCacheScope scope);

    [[nodiscard]] const std::type_info& getType() const;

    bool operator==(const PredictionCacheScopeTrait& other) const;

    [[nodiscard]] size_t hash() const;

    [[nodiscard]] std::string explain(ExplainVerbosity) const;

    [[nodiscard]] std::string_view getName() const;

    friend Reflector<PredictionCacheScopeTrait>;
};

template <>
struct Reflector<PredictionCacheScopeTrait>
{
    Reflected operator()(const PredictionCacheScopeTrait& trait) const;
};

template <>
struct Unreflector<PredictionCacheScopeTrait>
{
    PredictionCacheScopeTrait operator()(const Reflected& reflected) const;
};

static_assert(TraitConcept<PredictionCacheScopeTrait>);
}

namespace NES::detail
{
struct ReflectedPredictionCacheScopeTrait
{
    PredictionCacheScope scope;
};
}
