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
#include <string>
#include <string_view>
#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/FieldAccessLogicalFunction.hpp>
#include <Operators/Windows/Aggregations/WindowAggregationLogicalFunction.hpp>
#include <Util/Reflection.hpp>

namespace NES
{

class ArrayAggregationLogicalFunction
{
public:
    ArrayAggregationLogicalFunction(const FieldAccessLogicalFunction& onField, FieldAccessLogicalFunction asField);
    explicit ArrayAggregationLogicalFunction(const FieldAccessLogicalFunction& onField);
    ~ArrayAggregationLogicalFunction() = default;

    [[nodiscard]] std::string_view getName() const noexcept;
    [[nodiscard]] std::string toString() const;
    [[nodiscard]] Reflected reflect() const;
    [[nodiscard]] DataType getInputStamp() const;
    [[nodiscard]] DataType getPartialAggregateStamp() const;
    [[nodiscard]] DataType getFinalAggregateStamp() const;
    [[nodiscard]] FieldAccessLogicalFunction getOnField() const;
    [[nodiscard]] FieldAccessLogicalFunction getAsField() const;

    [[nodiscard]] ArrayAggregationLogicalFunction withInferredStamp(const Schema& schema) const;
    [[nodiscard]] ArrayAggregationLogicalFunction withInputStamp(DataType inputStamp) const;
    [[nodiscard]] ArrayAggregationLogicalFunction withPartialAggregateStamp(DataType partialAggregateStamp) const;
    [[nodiscard]] ArrayAggregationLogicalFunction withFinalAggregateStamp(DataType finalAggregateStamp) const;
    [[nodiscard]] ArrayAggregationLogicalFunction withOnField(FieldAccessLogicalFunction onField) const;
    [[nodiscard]] ArrayAggregationLogicalFunction withAsField(FieldAccessLogicalFunction asField) const;
    [[nodiscard]] static bool shallIncludeNullValues() noexcept;
    [[nodiscard]] static bool requiresSequentialAggregation() noexcept;
    [[nodiscard]] bool operator==(const ArrayAggregationLogicalFunction& other) const;

private:
    static constexpr std::string_view NAME = "ARRAY_AGG";

    DataType inputStamp;
    DataType partialAggregateStamp;
    DataType finalAggregateStamp;
    FieldAccessLogicalFunction onField;
    FieldAccessLogicalFunction asField;
};

static_assert(WindowAggregationFunctionConcept<ArrayAggregationLogicalFunction>);

template <>
struct Reflector<ArrayAggregationLogicalFunction>
{
    Reflected operator()(const ArrayAggregationLogicalFunction& function) const;
};

template <>
struct Unreflector<ArrayAggregationLogicalFunction>
{
    ArrayAggregationLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedArrayAggregationLogicalFunction
{
    FieldAccessLogicalFunction onField;
    FieldAccessLogicalFunction asField;
};
}
