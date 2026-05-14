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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/Enums/EnumOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/Validation/BooleanValidation.hpp>
#include <Configurations/Validation/NonZeroValidation.hpp>
#include <fmt/ranges.h>

namespace NES
{

enum class PredictionCacheType : uint8_t
{
    NONE,
    ALWAYS_MISS,
    TWO_QUEUES,
    FIFO,
    LFU,
    LRU,
    SECOND_CHANCE
};

class InferenceConfiguration final : public BaseConfiguration
{
public:
    InferenceConfiguration() = default;
    InferenceConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { }

    BoolOption useBatchDeduplication = {
        "use_batch_deduplication",
        "false",
        "Whether to invoke the model only on unique values of a batch",
        {std::make_shared<BooleanValidation>()}};
    EnumOption<PredictionCacheType> predictionCacheType = {
        "prediction_cache_type",
        PredictionCacheType::NONE,
        fmt::format("Type of prediction cache {}", fmt::join(magic_enum::enum_names<PredictionCacheType>(), ", "))};
    UIntOption numberOfEntriesPredictionCache = {
        "number_of_entries_prediction_cache",
        "1",
        "Size of the prediction cache",
        {std::make_shared<NonZeroValidation>()}};
    UIntOption openvinoInferenceNumThreads = {
        "openvino_inference_num_threads",
        "1",
        "OpenVINO ov::inference_num_threads",
        {std::make_shared<NonZeroValidation>()}};
    UIntOption openvinoNumStreams = {"openvino_num_streams", "0", "OpenVINO ov::num_streams"};
    BoolOption openvinoEnableCpuPinning = {
        "openvino_enable_cpu_pinning",
        "false",
        "OpenVINO ov::hint::enable_cpu_pinning",
        {std::make_shared<BooleanValidation>()}};

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {
            &useBatchDeduplication,
            &predictionCacheType,
            &numberOfEntriesPredictionCache,
            &openvinoInferenceNumThreads,
            &openvinoNumStreams,
            &openvinoEnableCpuPinning};
    }
};

}
