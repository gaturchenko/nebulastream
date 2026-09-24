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

#include <filesystem>
#include <optional>

#include <gtest/gtest.h>
#include <Inference.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>
#include <OpenVinoRuntimeBackend.hpp>

namespace NES
{
namespace
{

CompiledModel loadModel()
{
    auto imported = importModel(std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_1_to_1.onnx", ModelBackend::OPENVINO);
    EXPECT_TRUE(imported.has_value());
    auto compiled = compileModel(imported.value());
    EXPECT_TRUE(compiled.has_value());
    return compiled.value();
}

InferenceRuntimeOptions sharedOptions()
{
    return InferenceRuntimeOptions{.openvinoShareCompiledModel = true};
}

/// Sessions of one model share a single compiled model while they exist.
TEST(SharedCompiledModelTest, SessionsOfOneModelShareOneCompiledModel)
{
    const auto model = loadModel();
    const auto before = liveSharedCompiledModelCount();

    std::optional<InferenceRuntime> first{std::in_place};
    first->setup(model, 1, sharedOptions());
    std::optional<InferenceRuntime> second{std::in_place};
    second->setup(model, 1, sharedOptions());

    EXPECT_EQ(liveSharedCompiledModelCount(), before + 1);
}

/// The cache shares a compiled model among live sessions, it does not keep one alive: once the last session using it is gone, so
/// is the compiled model. Otherwise a model that was only compiled to measure its memory would stay resident forever.
TEST(SharedCompiledModelTest, CompiledModelIsReleasedWithItsLastSession)
{
    const auto model = loadModel();
    const auto before = liveSharedCompiledModelCount();

    {
        InferenceRuntime first;
        first.setup(model, 1, sharedOptions());
        InferenceRuntime second;
        second.setup(model, 1, sharedOptions());
        EXPECT_EQ(liveSharedCompiledModelCount(), before + 1);
    }

    EXPECT_EQ(liveSharedCompiledModelCount(), before);
}

/// A private session must not put anything into the shared cache.
TEST(SharedCompiledModelTest, PrivateSessionsDoNotShare)
{
    const auto model = loadModel();
    const auto before = liveSharedCompiledModelCount();

    InferenceRuntime first;
    first.setup(model, 1, InferenceRuntimeOptions{});
    InferenceRuntime second;
    second.setup(model, 1, InferenceRuntimeOptions{});

    EXPECT_EQ(liveSharedCompiledModelCount(), before);
}

}
}
