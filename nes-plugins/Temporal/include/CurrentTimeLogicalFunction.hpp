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

#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Util/Logger/Formatter.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <SerializableVariantDescriptor.pb.h>

namespace NES
{

/// Logical wall-clock time function.
/// Signature: CURRENT_TIME() -> UINT64  (microseconds since the Unix epoch, wall clock).
///
/// A nullary leaf function (no children); parsed via the generic zero-argument function-call path.
/// Its purpose is to stamp an ingestion time onto tuples (`SELECT *, CURRENT_TIME() AS ingestTime ...`)
/// so that per-record end-to-end latency can be measured downstream (e.g. by the Latency sink, which
/// subtracts this stamp from its own receive time).
///
/// NOTE: this is a NON-deterministic function — every evaluation reads the clock afresh. Use it exactly
/// once per intended timestamp; it is not meant to be common-subexpression-eliminated across two calls.
class CurrentTimeLogicalFunction final
{
public:
    static constexpr std::string_view NAME = "CURRENT_TIME";

    CurrentTimeLogicalFunction();

    [[nodiscard]] bool operator==(const CurrentTimeLogicalFunction& rhs) const;

    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] CurrentTimeLogicalFunction withDataType(const DataType& dataType) const;
    [[nodiscard]] LogicalFunction withInferredDataType(const Schema& schema) const;

    [[nodiscard]] std::vector<LogicalFunction> getChildren() const;
    [[nodiscard]] CurrentTimeLogicalFunction withChildren(const std::vector<LogicalFunction>& children) const;

    [[nodiscard]] std::string_view getType() const;
    [[nodiscard]] std::string explain(ExplainVerbosity verbosity) const;

private:
    DataType dataType;
};

static_assert(LogicalFunctionConcept<CurrentTimeLogicalFunction>);

template <>
struct Reflector<CurrentTimeLogicalFunction>
{
    Reflected operator()(const CurrentTimeLogicalFunction& function) const;
};

template <>
struct Unreflector<CurrentTimeLogicalFunction>
{
    CurrentTimeLogicalFunction operator()(const Reflected& reflected) const;
};

}

namespace NES::detail
{
struct ReflectedCurrentTimeLogicalFunction
{
    DataType dataType;
};
}

FMT_OSTREAM(NES::CurrentTimeLogicalFunction);
