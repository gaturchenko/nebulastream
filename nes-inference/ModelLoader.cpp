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

#include <DataTypes/DataTypeProvider.hpp>
#include <Model.hpp>
#include <ModelLoader.hpp>

#include <array>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <optional>
#include <ranges>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <Util/Common.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Strings.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/io.hpp>
#include <boost/process/v1/pipe.hpp>
#include <boost/process/v1/search_path.hpp>
#include <boost/process/v1/start_dir.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>

using namespace std::literals;

namespace NES::Nebuli::Inference
{

namespace
{
struct Tool
{
    Tool(const std::string_view& name, bool hasVersion) : name(name), hasVersion(hasVersion) { }
    std::string_view name;
    bool hasVersion = false;
    bool available = false;
    std::string version;
};

struct InferenceBackendAvailability
{
    bool iree = false;
    bool openVino = false;
};

const std::unordered_map<std::string, DataType>& backendDTypeMap()
{
    static const std::unordered_map<std::string, DataType> map = {
        {"UI8", DataTypeProvider::provideDataType(DataType::Type::UINT8)},
        {"UI16", DataTypeProvider::provideDataType(DataType::Type::UINT16)},
        {"UI32", DataTypeProvider::provideDataType(DataType::Type::UINT32)},
        {"UI64", DataTypeProvider::provideDataType(DataType::Type::UINT64)},
        {"I8", DataTypeProvider::provideDataType(DataType::Type::INT8)},
        {"I16", DataTypeProvider::provideDataType(DataType::Type::INT16)},
        {"I32", DataTypeProvider::provideDataType(DataType::Type::INT32)},
        {"I64", DataTypeProvider::provideDataType(DataType::Type::INT64)},
        {"F16", DataTypeProvider::provideDataType(DataType::Type::FLOAT32)},
        {"F32", DataTypeProvider::provideDataType(DataType::Type::FLOAT32)},
        {"F64", DataTypeProvider::provideDataType(DataType::Type::FLOAT64)},
        {"U8", DataTypeProvider::provideDataType(DataType::Type::UINT8)},
        {"U16", DataTypeProvider::provideDataType(DataType::Type::UINT16)},
        {"U32", DataTypeProvider::provideDataType(DataType::Type::UINT32)},
        {"U64", DataTypeProvider::provideDataType(DataType::Type::UINT64)},
        {"FP16", DataTypeProvider::provideDataType(DataType::Type::FLOAT32)},
        {"FP32", DataTypeProvider::provideDataType(DataType::Type::FLOAT32)},
        {"FP64", DataTypeProvider::provideDataType(DataType::Type::FLOAT64)},
    };
    return map;
}

DataType resolveDType(const std::string& serializedDType)
{
    const auto normalized = toUpperCase(serializedDType);
    if (const auto it = backendDTypeMap().find(normalized); it != backendDTypeMap().end())
    {
        return it->second;
    }
    return DataTypeProvider::provideDataType(DataType::Type::UNDEFINED);
}

struct ModelMetadataGraph
{
    struct VertexProps
    {
        std::string label;
        std::string shape;
    };

    struct ModelMetadata
    {
        std::vector<size_t> inputShape;
        std::string inputDtype;
        std::vector<size_t> outputShape;
        std::string outputDtype;
        std::string functionName;
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProps> Graph;
    Graph graph;

    ModelMetadataGraph(const std::string& dot_file_path)
    {
        boost::dynamic_properties dp = boost::dynamic_properties(boost::ignore_other_properties);
        dp.property("label", boost::get(&VertexProps::label, graph));
        dp.property("shape", boost::get(&VertexProps::shape, graph));

        std::ifstream in(dot_file_path);
        boost::read_graphviz(in, graph, dp);
    }

    std::vector<size_t> parseTensorShape(const std::string& label)
    {
        std::regex tensor_regex(R"(tensor<([?0-9x]+)(?:ui8|ui16|ui32|ui64|i8|i16|i32|i64|f32|f64)>)"); // NES-supported numeric types
        std::smatch match;
        std::vector<size_t> result;
        if (std::regex_search(label, match, tensor_regex))
        {
            std::string shape_str = match[1];
            std::stringstream ss(shape_str);
            std::string dim;
            while (std::getline(ss, dim, 'x'))
            {
                if (dim == "?")
                {
                    result.push_back(1);
                }
                else
                {
                    result.push_back(std::stoi(dim));
                }
            }
        }
        return result;
    }

    std::string parseTensorDtype(const std::string& label)
    {
        std::regex tensor_regex(R"(tensor<([?0-9x]+)x(ui8|ui16|ui32|ui64|i8|i16|i32|i64|f32|f64)>)"); // NES-supported numeric types
        std::smatch match;
        std::string dtype;
        if (std::regex_search(label, match, tensor_regex))
        {
            dtype = match[2];
        }
        return dtype;
    }

    std::string parseFunctionName(const std::string& label)
    {
        std::regex graph_name_regex(R"(@([a-zA-Z0-9_]+)\$)");
        std::smatch match;
        if (std::regex_search(label, match, graph_name_regex))
        {
            return match[1];
        }
        return {};
    }

    /// Function to get next node label if there's exactly one outgoing edge
    std::optional<std::string> findLabelOfUnambiguousSuccessor(auto v, const auto& g)
    {
        auto [ai, ai_end] = adjacent_vertices(v, g);
        /// Check if there's at least one adjacent vertex
        if (ai == ai_end)
        {
            return std::nullopt; // No outgoing edges
        }

        /// Check if there's more than one adjacent vertex
        auto next_vertex = *ai;
        ++ai;
        if (ai != ai_end)
        {
            return std::nullopt; // More than one outgoing edge
        }

        /// There's exactly one outgoing edge. Get the label of the target vertex
        return g[next_vertex].label;
    }

    ModelMetadata getModelMetadata()
    {
        ModelMetadata metadata;
        while (metadata.functionName.empty() || metadata.inputShape.empty() || metadata.outputShape.empty())
        {
            for (auto v : boost::make_iterator_range(boost::vertices(graph)))
            {
                const std::string& label = graph[v].label;

                if (label.find("hal.tensor.import") != std::string::npos)
                {
                    metadata.inputShape = parseTensorShape(label);
                    metadata.inputDtype = parseTensorDtype(label);
                }
                else if (label.find("flow.dispatch") != std::string::npos)
                {
                    metadata.functionName = parseFunctionName(label);
                }
                else if (findLabelOfUnambiguousSuccessor(v, graph)
                             .transform([](const auto& label) { return label.find("hal.tensor.export") != std::string::npos; })
                             .value_or(false))
                {
                    metadata.outputShape = parseTensorShape(label);
                    metadata.outputDtype = parseTensorDtype(label);
                }
            }
        }
        return metadata;
    }
};

struct OpenVinoModelMetadata
{
    std::vector<size_t> inputShape;
    std::string inputDtype;
    std::vector<size_t> outputShape;
    std::string outputDtype;
};

std::expected<std::vector<size_t>, std::string> parseShape(const std::string& serializedShape)
{
    std::vector<size_t> shape;
    std::stringstream shapeStream(serializedShape);
    std::string dim;
    while (std::getline(shapeStream, dim, ','))
    {
        const auto trimmed = trimWhiteSpaces(dim);
        if (trimmed.empty())
        {
            continue;
        }
        if (trimmed == "?" || trimmed == "-1")
        {
            shape.emplace_back(1);
            continue;
        }

        try
        {
            const auto parsed = std::stoll(std::string(trimmed));
            if (parsed < 0)
            {
                shape.emplace_back(1);
            }
            else
            {
                shape.emplace_back(static_cast<size_t>(parsed));
            }
        }
        catch (const std::exception& exception)
        {
            return std::unexpected(fmt::format("Failed to parse dimension '{}': {}", trimmed, exception.what()));
        }
    }
    if (shape.empty())
    {
        return std::unexpected("Model tensor shape is empty");
    }
    return shape;
}

pugi::xml_node findFirstDescendantByName(const pugi::xml_node& node, const char* targetName)
{
    for (const auto& child : node.children())
    {
        if (std::string_view{child.name()} == targetName)
        {
            return child;
        }
        if (const auto nested = findFirstDescendantByName(child, targetName); nested)
        {
            return nested;
        }
    }
    return {};
}

std::optional<std::string> findFirstAttributeRecursively(const pugi::xml_node& node, const char* attributeName)
{
    if (!node)
    {
        return std::nullopt;
    }

    if (const auto attribute = node.attribute(attributeName); attribute)
    {
        return attribute.value();
    }

    for (const auto& child : node.children())
    {
        if (const auto value = findFirstAttributeRecursively(child, attributeName); value.has_value())
        {
            return value;
        }
    }

    return std::nullopt;
}

std::expected<std::vector<size_t>, std::string> parseShapeFromLayer(const pugi::xml_node& layerNode)
{
    if (const auto shapeAttr = findFirstAttributeRecursively(layerNode, "shape"); shapeAttr.has_value())
    {
        return parseShape(*shapeAttr);
    }

    const auto firstPortNode = findFirstDescendantByName(layerNode, "port");
    if (!firstPortNode)
    {
        return std::unexpected("Could not find any <port> tag in model layer");
    }

    std::vector<size_t> shape;
    for (const auto& dimNode : firstPortNode.children("dim"))
    {
        const auto parsed = parseShape(dimNode.text().as_string());
        if (!parsed)
        {
            return std::unexpected(parsed.error());
        }
        shape.insert(shape.end(), parsed->begin(), parsed->end());
    }

    if (shape.empty())
    {
        return std::unexpected("Could not parse model shape from layer");
    }
    return shape;
}

std::string parseDTypeFromLayer(const pugi::xml_node& layerNode)
{
    if (const auto elementType = findFirstAttributeRecursively(layerNode, "element_type"); elementType.has_value())
    {
        return *elementType;
    }
    if (const auto precision = findFirstAttributeRecursively(layerNode, "precision"); precision.has_value())
    {
        return *precision;
    }
    return {};
}

std::expected<OpenVinoModelMetadata, std::string> parseOpenVinoMetadata(const std::string& xmlContent)
{
    pugi::xml_document document;
    const auto parseResult = document.load_string(xmlContent.c_str());
    if (!parseResult)
    {
        return std::unexpected(fmt::format("Could not parse OpenVINO XML: {}", parseResult.description()));
    }

    const auto layersNode = findFirstDescendantByName(document, "layers");
    if (!layersNode)
    {
        return std::unexpected("Could not find <layers> in OpenVINO XML");
    }

    std::vector<pugi::xml_node> layers;
    for (const auto& child : layersNode.children("layer"))
    {
        layers.emplace_back(child);
    }

    if (layers.empty())
    {
        return std::unexpected("OpenVINO XML does not contain any <layer> entries");
    }

    OpenVinoModelMetadata metadata;
    const auto parsedInputShape = parseShapeFromLayer(layers.front());
    if (!parsedInputShape)
    {
        return std::unexpected(fmt::format("Failed to parse input shape: {}", parsedInputShape.error()));
    }
    metadata.inputShape = *parsedInputShape;
    metadata.inputDtype = parseDTypeFromLayer(layers.front());

    const auto parsedOutputShape = parseShapeFromLayer(layers.back());
    if (!parsedOutputShape)
    {
        return std::unexpected(fmt::format("Failed to parse output shape: {}", parsedOutputShape.error()));
    }
    metadata.outputShape = *parsedOutputShape;
    metadata.outputDtype = parseDTypeFromLayer(layers.back());

    return metadata;
}

auto format_as(const Tool& tool)
{
    if (tool.available)
    {
        if (tool.hasVersion)
        {
            return fmt::format("{}: {}", tool.name, tool.version);
        }
        else
        {
            return fmt::format("{}: available", tool.name);
        }
    }
    else
    {
        return fmt::format("{}: Not Found", tool.name);
    }
}

InferenceBackendAvailability checkInferenceToolsAreAvailable()
{
    std::array tools = {Tool{"iree-import-onnx"sv, false}, Tool{"iree-compile"sv, true}, Tool{"ovc"sv, true}};
    for (auto& tool : tools)
    {
        auto binaryInPath = boost::process::v1::search_path(std::string(tool.name));
        if (binaryInPath.empty())
        {
            NES_WARNING("{} is not in PATH", tool.name);
            continue;
        }
        tool.available = true;

        if (tool.hasVersion)
        {
            try
            {
                // Create a child process that executes 'command --version'
                // Redirect stdout to null to avoid printing output
                boost::process::v1::ipstream pipe_stream;
                boost::process::v1::child c(binaryInPath, "--version", boost::process::v1::std_out > pipe_stream);

                // Read the output
                std::string line;
                while (pipe_stream && std::getline(pipe_stream, line))
                {
                    tool.version += line + "\n";
                }

                // Wait for the process to finish
                c.wait();
            }
            catch (const boost::process::v1::process_error& bpe)
            {
                NES_WARNING("Could not retrieve version of '{}':\n{}", tool.name, bpe.what());
                tool.available = false;
            }
        }
    }

    if (!std::ranges::all_of(tools, &Tool::available))
    {
        NES_WARNING(
            "Missing Inference Tools:\n{}", fmt::join(std::views::filter(tools, [](const auto& tool) { return !tool.available; }), "\n"));
    }
    NES_INFO("Installed Inference Tools:\n{}", fmt::join(std::views::filter(tools, &Tool::available), "\n"));

    const auto isAvailable = [&](const std::string_view toolName)
    {
        const auto toolIt = std::ranges::find_if(tools, [&](const auto& tool) { return tool.name == toolName; });
        return toolIt != tools.end() && toolIt->available;
    };
    return InferenceBackendAvailability{
        .iree = isAvailable("iree-import-onnx"sv) && isAvailable("iree-compile"sv),
        .openVino = isAvailable("ovc"sv)};
}

const InferenceBackendAvailability& getInferenceBackendAvailability()
{
    static const InferenceBackendAvailability availability = checkInferenceToolsAreAvailable();
    return availability;
}

bool isBackendAvailable(const ModelBackend backend)
{
    const auto& availability = getInferenceBackendAvailability();
    switch (backend)
    {
        case ModelBackend::IREE: return availability.iree;
        case ModelBackend::OPENVINO: return availability.openVino;
    }
    std::unreachable();
}

std::expected<std::vector<std::byte>, ModelLoadError> readBinaryFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.good())
    {
        return std::unexpected(ModelLoadError{fmt::format("Could not open `{}`", filePath.string())});
    }

    file.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> bytes(size);
    if (size > 0)
    {
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    }
    return bytes;
}

std::expected<std::string, ModelLoadError> readTextFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath);
    if (!file.good())
    {
        return std::unexpected(ModelLoadError{fmt::format("Could not open `{}`", filePath.string())});
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::optional<std::filesystem::path> findGeneratedFile(const std::filesystem::path& directory, const std::string_view extension)
{
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == extension)
        {
            return entry.path();
        }
    }
    return std::nullopt;
}

std::expected<Model, ModelLoadError> loadIreeModel(const std::filesystem::path& modelPath, const ModelOptions& options)
{
    std::vector<std::string> importArgs;
    std::vector<std::string> compileArgs;

    importArgs.emplace_back(modelPath);

    if (auto opset = options.opset)
    {
        importArgs.emplace_back("--opset-version");
        importArgs.emplace_back(std::to_string(*opset));
    }

    /// Pipe output of import to compiler
    compileArgs.emplace_back("-");

    /// target hardware info
    compileArgs.emplace_back("--iree-hal-target-device=local");
    compileArgs.emplace_back("--iree-hal-local-target-device-backends=llvm-cpu");
    ///TODO: (#???) This only works if nebuli and the worker are running on the same arch
    compileArgs.emplace_back("--iree-llvmcpu-target-cpu=host");

    /// reducing the artifact size
    compileArgs.emplace_back("--iree-llvmcpu-debug-symbols=false");
    compileArgs.emplace_back("--iree-llvmcpu-keep-linker-artifacts=false");

    /// optimizations
    /// we want to optimize for lower binary size and instruction latency
    compileArgs.emplace_back("--cost-kind=size-latency");
    compileArgs.emplace_back("--iree-opt-level=O2");
    compileArgs.emplace_back("--iree-opt-data-tiling");
    /// we run IREE single-threaded so we can optimize for minimum peak memory usage with no sacrifices
    compileArgs.emplace_back("--iree-stream-partitioning-favor=min-peak-memory");

    /// iree-compile allows to dump a .dot graph containing dispatch operations
    /// while this is not exactly metadata, we can still extract the necessary information from it
    auto tempPath = createTempDir("/tmp/nebuli-model-loader");
    TempDirectoryCleanup removeTempPath{tempPath};

    auto graphFile = tempPath / "model.dot";

    compileArgs.emplace_back("--iree-flow-dump-dispatch-graph");
    compileArgs.emplace_back(fmt::format("--iree-flow-dump-dispatch-graph-output-file={}", graphFile));

    try
    {
        boost::process::v1::pipe mlirPipe;
        boost::process::v1::ipstream importError;
        boost::process::v1::ipstream compileError;
        boost::process::v1::ipstream modelStream;
        std::vector<boost::process::v1::child> process;
        process.emplace_back(
            boost::process::v1::search_path("iree-import-onnx"),
            importArgs,
            boost::process::v1::std_out > mlirPipe,
            boost::process::v1::std_err > importError);
        process.emplace_back(
            boost::process::v1::search_path("iree-compile"),
            compileArgs,
            boost::process::v1::std_in<mlirPipe, boost::process::v1::std_out> modelStream,
            boost::process::v1::std_err > compileError);

        /// Read output of iree-compile into a byte buffer
        std::vector<std::byte> modelVmfb;
        std::array<std::byte, 4096> buffer;

        while (process[1].running() && modelStream.good())
        {
            modelStream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

            const std::streamsize count = modelStream.gcount();
            if (count <= 0)
            {
                break;
            }

            const std::span bytesWritten = {buffer.data(), static_cast<size_t>(count)};
            std::ranges::copy(bytesWritten, std::back_inserter(modelVmfb));
        }

        std::ranges::for_each(process, [](auto& childProcess) { childProcess.wait(); });
        const auto success = std::ranges::all_of(process, [](const auto& childProcess) { return childProcess.exit_code() == 0; });
        if (success)
        {
            auto modelVmfbBuffer = std::make_shared<std::byte[]>(modelVmfb.size());
            std::ranges::copy(modelVmfb, modelVmfbBuffer.get());

            ModelMetadataGraph modelGraph(graphFile);
            auto metadata = modelGraph.getModelMetadata();

            Model model = Model{std::move(modelVmfbBuffer), modelVmfb.size()};
            model.setFunctionName("module." + metadata.functionName);
            model.setInputShape(metadata.inputShape);
            model.setOutputShape(metadata.outputShape);
            model.setInputDtype(resolveDType(metadata.inputDtype));
            model.setOutputDtype(resolveDType(metadata.outputDtype));
            return model;
        }
        NES_ERROR(
            "Errors during Model Import:\nIree Import Error:\n{} {}\n```\n{}```\nIree Compile Error:\n{} {}\n```\n{}```",
            boost::process::v1::search_path("iree-import-onnx").string(),
            fmt::join(importArgs, " "),
            std::string{std::istreambuf_iterator(importError), std::istreambuf_iterator<char>()},
            boost::process::v1::search_path("iree-compile").string(),
            fmt::join(compileArgs, " "),
            std::string{std::istreambuf_iterator(compileError), std::istreambuf_iterator<char>()});
        return std::unexpected(ModelLoadError("Model import was not successful: non-zero exit code."));
    }
    catch (const boost::process::v1::process_error& bpe)
    {
        return std::unexpected(ModelLoadError(fmt::format("Model import was not successful: {}", bpe.what())));
    }
}

std::expected<Model, ModelLoadError> loadOpenVinoModel(const std::filesystem::path& modelPath)
{
    auto tempPath = createTempDir("/tmp/nebuli-model-loader-openvino");
    TempDirectoryCleanup removeTempPath{tempPath};

    std::vector<std::string> ovcArgs = {modelPath.string(), "--compress_to_fp16=False"};
    boost::process::v1::ipstream ovcError;
    try
    {
        boost::process::v1::child process(
            boost::process::v1::search_path("ovc"),
            ovcArgs,
            boost::process::v1::start_dir = tempPath,
            boost::process::v1::std_err > ovcError);
        process.wait();
        if (process.exit_code() != 0)
        {
            return std::unexpected(ModelLoadError(
                fmt::format(
                    "OpenVINO conversion failed ({}): {}",
                    process.exit_code(),
                    std::string{std::istreambuf_iterator(ovcError), std::istreambuf_iterator<char>()})));
        }
    }
    catch (const boost::process::v1::process_error& error)
    {
        return std::unexpected(ModelLoadError(fmt::format("OpenVINO conversion failed: {}", error.what())));
    }

    auto xmlPath = tempPath / fmt::format("{}.xml", modelPath.stem().string());
    auto binPath = tempPath / fmt::format("{}.bin", modelPath.stem().string());
    if (!std::filesystem::exists(xmlPath))
    {
        xmlPath = findGeneratedFile(tempPath, ".xml").value_or(xmlPath);
    }
    if (!std::filesystem::exists(binPath))
    {
        binPath = findGeneratedFile(tempPath, ".bin").value_or(binPath);
    }

    if (!std::filesystem::exists(xmlPath) || !std::filesystem::exists(binPath))
    {
        return std::unexpected(ModelLoadError(
            fmt::format("OpenVINO conversion did not produce both XML and BIN files under `{}`", tempPath.string())));
    }

    auto xmlContentResult = readTextFile(xmlPath);
    if (!xmlContentResult)
    {
        return std::unexpected(xmlContentResult.error());
    }
    auto binContentResult = readBinaryFile(binPath);
    if (!binContentResult)
    {
        return std::unexpected(binContentResult.error());
    }

    auto metadataResult = parseOpenVinoMetadata(*xmlContentResult);
    if (!metadataResult)
    {
        return std::unexpected(ModelLoadError(fmt::format("Failed to parse OpenVINO metadata: {}", metadataResult.error())));
    }

    auto xmlBuffer = std::make_shared<std::byte[]>(xmlContentResult->size());
    std::ranges::copy(
        *xmlContentResult | std::views::transform([](const auto& character) { return static_cast<std::byte>(character); }),
        xmlBuffer.get());

    auto binBuffer = std::make_shared<std::byte[]>(binContentResult->size());
    std::ranges::copy(*binContentResult, binBuffer.get());

    Model model{xmlBuffer, xmlContentResult->size(), binBuffer, binContentResult->size()};
    model.setInputShape(metadataResult->inputShape);
    model.setOutputShape(metadataResult->outputShape);
    model.setInputDtype(resolveDType(metadataResult->inputDtype));
    model.setOutputDtype(resolveDType(metadataResult->outputDtype));
    return model;
}

}

bool enabled()
{
#ifndef NEBULI_INFERENCE_SUPPORT
    return false;
#else
    const auto& availability = getInferenceBackendAvailability();
    return availability.iree || availability.openVino;
#endif
}

std::expected<Model, ModelLoadError> load(const std::filesystem::path& modelPath, const ModelOptions& options)
{
    if (modelPath.filename().extension() != ".onnx")
    {
        return std::unexpected(ModelLoadError{"Loading does only support `.onnx` models at the moment"});
    }

    if (!std::filesystem::exists(modelPath))
    {
        return std::unexpected(ModelLoadError{fmt::format("Model `{}` does not exist.", modelPath.string())});
    }

    if (!isBackendAvailable(options.backend))
    {
        return std::unexpected(ModelLoadError(fmt::format(
            "Requested backend `{}` is not available in PATH",
            modelBackendToString(options.backend))));
    }

    switch (options.backend)
    {
        case ModelBackend::IREE: return loadIreeModel(modelPath, options);
        case ModelBackend::OPENVINO: return loadOpenVinoModel(modelPath);
    }
    std::unreachable();
}
}
