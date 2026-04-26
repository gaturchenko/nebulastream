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
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <Util/Logger/Logger.hpp>
#include <fmt/ranges.h>

#include <SerializableVariantDescriptor.pb.h>
#include <DataTypes/DataType.hpp>

namespace NES::Nebuli::Inference
{
struct ModelLoadError;
struct ModelOptions;

enum class ModelBackend : uint8_t
{
    IREE = 0,
    OPENVINO = 1,
};

std::optional<ModelBackend> parseModelBackend(std::string_view backend);
std::string_view modelBackendToString(ModelBackend backend);

class Model
{
    struct RefCountedByteBuffer
    {
        std::shared_ptr<std::byte[]> buffer;
        size_t size = 0;
        friend bool operator==(const RefCountedByteBuffer& lhs, const RefCountedByteBuffer& rhs)
        {
            return std::ranges::equal(std::span{lhs.buffer.get(), lhs.size}, std::span{rhs.buffer.get(), rhs.size});
        }
        friend bool operator!=(const RefCountedByteBuffer& lhs, const RefCountedByteBuffer& rhs) { return !(lhs == rhs); }
        std::span<const std::byte> getBuffer() const { return {buffer.get(), size}; }
    };

    ModelBackend backend = ModelBackend::OPENVINO;
    RefCountedByteBuffer byteCode;
    RefCountedByteBuffer openVinoXml;
    RefCountedByteBuffer openVinoBin;
    mutable std::vector<size_t> inputShape;
    std::vector<size_t> outputShape;
    std::string functionName;
    size_t inputDims = 0;
    size_t outputDims = 0;
    size_t inputSizeInBytes = 0;
    size_t outputSizeInBytes = 0;
    std::vector<DataType> inputs;
    std::vector<std::pair<std::string, DataType>> outputs;
    DataType inputDtype;
    DataType outputDtype;

public:
    Model(std::shared_ptr<std::byte[]> modelByteCode, size_t modelSize)
        : backend(ModelBackend::IREE), byteCode(std::move(modelByteCode), modelSize)
    {
    }

    Model(std::shared_ptr<std::byte[]> xmlBuffer, size_t xmlSize, std::shared_ptr<std::byte[]> binBuffer, size_t binSize)
        : backend(ModelBackend::OPENVINO), openVinoXml(std::move(xmlBuffer), xmlSize), openVinoBin(std::move(binBuffer), binSize)
    {
    }

    [[nodiscard]] ModelBackend getBackend() const { return backend; }
    std::span<const std::byte> getByteCode() const { return byteCode.getBuffer(); }
    std::span<const std::byte> getOpenVinoXml() const { return openVinoXml.getBuffer(); }
    std::span<const std::byte> getOpenVinoBin() const { return openVinoBin.getBuffer(); }
    void setFunctionName(std::string name) { functionName = std::move(name); }
    void setInputShape(std::vector<size_t> shape) { inputShape = std::move(shape); }
    void setOutputShape(std::vector<size_t> shape) { outputShape = std::move(shape); }
    void setInputDtype(const DataType dtype) { inputDtype = dtype; }
    void setOutputDtype(const DataType dtype) { outputDtype = dtype; }
    const std::vector<DataType>& getInputs() const { return inputs; }
    const std::vector<std::pair<std::string, DataType>>& getOutputs() const { return outputs; }

    bool operator==(const Model&) const = default;

    const std::vector<size_t>& getInputShape() const { return inputShape; }
    const std::vector<size_t>& getOutputShape() const { return outputShape; }

    size_t getNDim() { return inputDims; }
    size_t getOutputDims() { return outputDims; }

    size_t inputSize() const { return inputSizeInBytes; }
    size_t outputSize() const { return outputSizeInBytes; }

    const std::string& getFunctionName() { return functionName; }

    DataType getInputDtype() const { return inputDtype; }
    DataType getOutputDtype() const { return outputDtype; }

    friend class ModelCatalog;
    friend std::expected<Model, ModelLoadError> load(const std::filesystem::path& path, const ModelOptions& options);
    friend Model deserializeModel(const SerializableModel& grpcModel);
    friend void serializeModel(const Model& model, SerializableModel& target);
};

Model deserializeModel(const SerializableModel& grpcModel);
void serializeModel(const Model& model, SerializableModel& target);
}
