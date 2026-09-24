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

#include <Traits/InferenceBatchSizeTrait.hpp>

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
#include <TraitRegisty.hpp>

namespace NES
{
/// Required for plugin registration, no implementation necessary
TraitRegistryReturnType TraitGeneratedRegistrar::RegisterInferenceBatchSizeTrait(TraitRegistryArguments arguments)
{
    return unreflect<InferenceBatchSizeTrait>(arguments.reflected);
}

InferenceBatchSizeTrait::InferenceBatchSizeTrait(const uint64_t batchSize) : batchSize(batchSize)
{
}

const std::type_info& InferenceBatchSizeTrait::getType() const
{
    return typeid(InferenceBatchSizeTrait);
}

bool InferenceBatchSizeTrait::operator==(const InferenceBatchSizeTrait& other) const
{
    return batchSize == other.batchSize;
}

size_t InferenceBatchSizeTrait::hash() const
{
    return std::hash<uint64_t>{}(batchSize);
}

std::string InferenceBatchSizeTrait::explain(ExplainVerbosity) const
{
    return fmt::format("InferenceBatchSizeTrait: {}", batchSize);
}

std::string_view InferenceBatchSizeTrait::getName() const
{
    return NAME;
}

Reflected Reflector<InferenceBatchSizeTrait>::operator()(const InferenceBatchSizeTrait& trait) const
{
    return reflect(detail::ReflectedInferenceBatchSizeTrait{trait.batchSize});
}

InferenceBatchSizeTrait Unreflector<InferenceBatchSizeTrait>::operator()(const Reflected& reflected) const
{
    auto [batchSize] = unreflect<detail::ReflectedInferenceBatchSizeTrait>(reflected);
    return InferenceBatchSizeTrait{batchSize};
}

}
