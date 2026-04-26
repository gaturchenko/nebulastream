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

#include <boost/fusion/sequence/io/out.hpp>


#include <Serialization/DataTypeSerializationUtil.hpp>
#include <Model.hpp>
#include <Util/Strings.hpp>
#include <ranges>
#include <utility>

namespace
{
std::pair<std::shared_ptr<std::byte[]>, size_t> copyToByteBuffer(const std::string& source)
{
    const auto size = source.size();
    auto buffer = std::make_shared<std::byte[]>(size);
    std::ranges::copy(
        source | std::views::transform([](const auto& character) { return static_cast<std::byte>(character); }),
        buffer.get());
    return {std::move(buffer), size};
}
}

std::optional<NES::Nebuli::Inference::ModelBackend> NES::Nebuli::Inference::parseModelBackend(const std::string_view backend)
{
    const auto normalized = toUpperCase(backend);
    if (normalized == "IREE")
    {
        return ModelBackend::IREE;
    }
    if (normalized == "OPENVINO")
    {
        return ModelBackend::OPENVINO;
    }
    return std::nullopt;
}

std::string_view NES::Nebuli::Inference::modelBackendToString(const ModelBackend backend)
{
    switch (backend)
    {
        case ModelBackend::IREE: return "IREE";
        case ModelBackend::OPENVINO: return "OPENVINO";
    }
    std::unreachable();
}

NES::Nebuli::Inference::Model NES::Nebuli::Inference::deserializeModel(const SerializableModel& grpcModel)
{
    ModelBackend backend = ModelBackend::OPENVINO;
    switch (grpcModel.backend())
    {
        case SerializableModel_ModelBackend_IREE: backend = ModelBackend::IREE; break;
        case SerializableModel_ModelBackend_OPENVINO: backend = ModelBackend::OPENVINO; break;
        default:
            backend = ModelBackend::OPENVINO;
            break;
    }

    Model model = [&]()
    {
        if (backend == ModelBackend::OPENVINO)
        {
            const auto [xmlBuffer, xmlSize] = copyToByteBuffer(grpcModel.openvinoxml());
            const auto [binBuffer, binSize] = copyToByteBuffer(grpcModel.openvinobin());
            return Model{xmlBuffer, xmlSize, binBuffer, binSize};
        }
        const auto [byteCodeBuffer, byteCodeSize] = copyToByteBuffer(grpcModel.bytecode());
        return Model{byteCodeBuffer, byteCodeSize};
    }();

    model.backend = backend;
    model.functionName = grpcModel.functionname();
    model.inputDims = grpcModel.dims();
    model.inputShape.assign(grpcModel.shape().begin(), grpcModel.shape().end());

    model.outputDims = grpcModel.outputdims();
    model.outputShape.assign(grpcModel.outputshape().begin(), grpcModel.outputshape().end());

    model.inputSizeInBytes = grpcModel.inputsizeinbytes();
    model.outputSizeInBytes = grpcModel.outputsizeinbytes();
    model.inputs = grpcModel.inputs()
        | std::views::transform([](const auto& serializedDataType)
                                { return DataTypeSerializationUtil::deserializeDataType(serializedDataType); })
        | std::ranges::to<std::vector>();

    model.outputs = grpcModel.outputs()
        | std::views::transform(
                        [](const auto& typeWithName)
                        {
                            return std::pair<std::string, DataType>{
                                typeWithName.name(), DataTypeSerializationUtil::deserializeDataType(typeWithName.type())};
                        })
        | std::ranges::to<std::vector>();

    model.inputDtype = DataTypeSerializationUtil::deserializeDataType(grpcModel.inputdtype());
    model.outputDtype = DataTypeSerializationUtil::deserializeDataType(grpcModel.outputdtype());

    return model;
}

void NES::Nebuli::Inference::serializeModel(const Model& model, SerializableModel& target)
{
    switch (model.backend)
    {
        case ModelBackend::IREE: {
            auto modelBytes = model.getByteCode() | std::views::transform([](const std::byte& byte) { return static_cast<const char>(byte); });
            target.mutable_bytecode()->assign(modelBytes.begin(), modelBytes.end());
            break;
        }
        case ModelBackend::OPENVINO: {
            auto xmlBytes = model.getOpenVinoXml() | std::views::transform([](const std::byte& byte) { return static_cast<const char>(byte); });
            target.mutable_openvinoxml()->assign(xmlBytes.begin(), xmlBytes.end());
            auto binBytes = model.getOpenVinoBin() | std::views::transform([](const std::byte& byte) { return static_cast<const char>(byte); });
            target.mutable_openvinobin()->assign(binBytes.begin(), binBytes.end());
            break;
        }
    }

    switch (model.backend)
    {
        case ModelBackend::IREE: target.set_backend(SerializableModel_ModelBackend_IREE); break;
        case ModelBackend::OPENVINO: target.set_backend(SerializableModel_ModelBackend_OPENVINO); break;
    }

    target.set_dims(model.inputDims);
    for (int shape : model.inputShape)
    {
        target.add_shape(shape);
    }

    target.set_outputdims(model.outputDims);
    for (int shape : model.outputShape)
    {
        target.add_outputshape(shape);
    }

    target.set_functionname(model.functionName);
    target.set_inputsizeinbytes(model.inputSizeInBytes);
    target.set_outputsizeinbytes(model.outputSizeInBytes);
    for (auto& input : model.inputs)
    {
        DataTypeSerializationUtil::serializeDataType(input, target.add_inputs());
    }

    for (auto& [name, type] : model.outputs)
    {
        auto* output = target.add_outputs();
        output->set_name(name);
        DataTypeSerializationUtil::serializeDataType(type, output->mutable_type());
    }

    DataTypeSerializationUtil::serializeDataType(model.inputDtype, target.mutable_inputdtype());
    DataTypeSerializationUtil::serializeDataType(model.outputDtype, target.mutable_outputdtype());
}
