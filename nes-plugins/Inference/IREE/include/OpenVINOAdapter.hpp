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
#include <memory>
#include <ostream>
#include <span>
#include <unordered_map>
#include <DataTypes/DataTypeProvider.hpp>
#include <ErrorHandling.hpp>
#include <Model.hpp>
#include <OpenVINORuntimeWrapper.hpp>

namespace NES
{

class OpenVINOAdapter
{
public:
    static std::shared_ptr<OpenVINOAdapter> create();

    OpenVINOAdapter() = default;

    void initializeModel(Nebuli::Inference::Model& model, uint64_t batchSize);

    template <class T>
    void addModelInput(size_t index, T value);
    void addModelInput(std::span<std::byte> content);

    template <class T>
    T getResultAt(size_t idx);
    void copyResultTo(std::span<std::byte> content);

    template <class T>
    void infer();

    std::unique_ptr<std::byte[]> inputData{};
    std::unique_ptr<std::byte[]> outputData{};

    size_t inputSize = 0;
    size_t outputSize = 0;

private:
    OpenVINORuntimeWrapper runtimeWrapper;
    std::unordered_map<NES::DataType, ov::element::Type> dtypeMap = {
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT8), ov::element::u8},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT16), ov::element::u16},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT32), ov::element::u32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT64), ov::element::u64},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT8), ov::element::i8},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT16), ov::element::i16},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT32), ov::element::i32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT64), ov::element::i64},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::FLOAT32), ov::element::f32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::FLOAT64), ov::element::f64},
    };
};

}
