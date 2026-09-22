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

#include <chrono>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Phases/InferenceImplementationSelector.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/PlacementTrait.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <ErrorHandling.hpp>
#include <InferenceProfiler.hpp>
#include <ModelCatalog.hpp>
#include <ModelProfile.hpp>
#include <QueryId.hpp>
#include <QueryOptimizerConfiguration.hpp>

namespace NES
{
namespace
{

/// Answers profiling jobs with fixed per-invocation latencies and records the submitted jobs.
class FakeInferenceProfiler final : public InferenceProfiler
{
public:
    FakeInferenceProfiler(std::chrono::nanoseconds singleTupleLatency, std::chrono::nanoseconds batchedLatency)
        : singleTupleLatency(singleTupleLatency), batchedLatency(batchedLatency)
    {
    }

    std::expected<ModelProfile, Exception> profile(const Host& worker, const ModelProfilingRequest& request) override
    {
        jobs.emplace_back(worker, request.batchSizes);
        if (failing)
        {
            return std::unexpected(InferenceRuntimeFailure("profiling failed"));
        }
        ModelProfile profile;
        for (const auto batchSize : request.batchSizes)
        {
            const auto latency = batchSize == 1 ? singleTupleLatency : batchedLatency;
            profile.batches.push_back(BatchProfile{
                .batchSize = batchSize,
                .iterations = request.measurementIterations,
                .medianLatency = latency,
                .meanLatency = latency,
                .minLatency = latency});
        }
        return profile;
    }

    std::chrono::nanoseconds singleTupleLatency;
    std::chrono::nanoseconds batchedLatency;
    bool failing = false;
    std::vector<std::pair<Host, std::vector<size_t>>> jobs;
};

class InferenceImplementationSelectorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("InferenceImplementationSelectorTest.log", LogLevel::LOG_DEBUG); }

    static inline const Host worker{"worker-1:8080"};

    static RegisteredModel loadVarsizedModel()
    {
        static ModelCatalog catalog;
        if (!catalog.hasModel("varsized"))
        {
            catalog.registerModel(
                "varsized",
                std::filesystem::path(INFERENCE_TEST_DATA) / "tiny_1_to_1.onnx",
                ModelSchema{
                    .inputs = Schema{}.addField("pixels", DataType::Type::VARSIZED),
                    .outputs = Schema{}.addField("prediction", DataType::Type::FLOAT32)});
        }
        return catalog.load("varsized");
    }

    /// A placed inference operator over the payload fields of a joined record.
    static LogicalPlan createPostJoinInferencePlan(const std::vector<std::string>& payloadFields, const std::optional<Host>& placement)
    {
        auto inferModel = InferModelLogicalOperator(loadVarsizedModel(), payloadFields);
        if (placement.has_value())
        {
            TraitSet traitSet;
            tryInsert(traitSet, PlacementTrait{placement.value()});
            inferModel = inferModel.withTraitSet(traitSet);
        }
        return LogicalPlan{INVALID_QUERY_ID, {LogicalOperator{inferModel}}};
    }

    static QueryOptimizerConfiguration createConfiguration(const PostJoinInferenceStrategy strategy)
    {
        QueryOptimizerConfiguration configuration;
        configuration.postJoinInferenceStrategy.setValue(strategy);
        return configuration;
    }

    static std::optional<PostJoinInferenceImplementation> getSelectedImplementation(const LogicalPlan& plan)
    {
        const auto trait = getTrait<PostJoinInferenceImplementationTrait>(plan.getRootOperators().front().getTraitSet());
        if (!trait.has_value())
        {
            return std::nullopt;
        }
        return trait.value()->implementation;
    }
};

TEST_F(InferenceImplementationSelectorTest, BatchesWhenBatchInvocationAmortizes)
{
    /// Two single-tuple invocations take 2 x 100 ns, one batch-2 invocation 120 ns: speedup 1.67
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});
    const InferenceImplementationSelector selector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};

    const auto plan = selector.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker));

    EXPECT_EQ(getSelectedImplementation(plan), PostJoinInferenceImplementation::BATCHED);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().first, worker);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2}));
}

TEST_F(InferenceImplementationSelectorTest, InvokesSequentiallyWithoutAmortization)
{
    /// Two single-tuple invocations take 2 x 100 ns, one batch-2 invocation 195 ns: speedup 1.03 is below the default 1.1
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{195});
    const InferenceImplementationSelector selector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};

    const auto plan = selector.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker));

    EXPECT_EQ(getSelectedImplementation(plan), PostJoinInferenceImplementation::SEQUENTIAL);
}

TEST_F(InferenceImplementationSelectorTest, ComparesAgainstNumberOfJoinedPayloads)
{
    /// Three single-tuple invocations take 3 x 100 ns, one batch-3 invocation 250 ns: speedup 1.2
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{250});
    const InferenceImplementationSelector selector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};

    const auto plan = selector.apply(createPostJoinInferencePlan({"s1$pixels", "s2$pixels", "s3$pixels"}, worker));

    EXPECT_EQ(getSelectedImplementation(plan), PostJoinInferenceImplementation::BATCHED);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 3}));
}

TEST_F(InferenceImplementationSelectorTest, RespectsConfiguredMinimumSpeedup)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});
    auto configuration = createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES);
    configuration.postJoinInferenceMinimumSpeedup.setValue(2.0F);
    const InferenceImplementationSelector selector{configuration, profiler};

    const auto plan = selector.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker));

    EXPECT_EQ(getSelectedImplementation(plan), PostJoinInferenceImplementation::SEQUENTIAL);
}

TEST_F(InferenceImplementationSelectorTest, ReusesProfilesOfTheSameModelOnTheSameWorker)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});
    const InferenceImplementationSelector selector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};

    (void)selector.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker));
    (void)selector.apply(createPostJoinInferencePlan({"stream3$pixels", "stream4$pixels"}, worker));
    EXPECT_EQ(profiler->jobs.size(), 1);

    (void)selector.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, Host{"worker-2:8080"}));
    EXPECT_EQ(profiler->jobs.size(), 2);
}

TEST_F(InferenceImplementationSelectorTest, FixedStrategiesDoNotProfile)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});

    const InferenceImplementationSelector batched{createConfiguration(PostJoinInferenceStrategy::BATCHED), profiler};
    EXPECT_EQ(
        getSelectedImplementation(batched.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker))),
        PostJoinInferenceImplementation::BATCHED);

    const InferenceImplementationSelector sequential{createConfiguration(PostJoinInferenceStrategy::SEQUENTIAL), profiler};
    EXPECT_EQ(
        getSelectedImplementation(sequential.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker))),
        PostJoinInferenceImplementation::SEQUENTIAL);

    EXPECT_TRUE(profiler->jobs.empty());
}

TEST_F(InferenceImplementationSelectorTest, FallsBackToBatchingIfProfilingIsImpossible)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{195});
    profiler->failing = true;
    const InferenceImplementationSelector failingProfiler{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};
    EXPECT_EQ(
        getSelectedImplementation(failingProfiler.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker))),
        PostJoinInferenceImplementation::BATCHED);

    const InferenceImplementationSelector noProfiler{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), nullptr};
    EXPECT_EQ(
        getSelectedImplementation(noProfiler.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, worker))),
        PostJoinInferenceImplementation::BATCHED);

    profiler->failing = false;
    const InferenceImplementationSelector unplaced{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};
    EXPECT_EQ(
        getSelectedImplementation(unplaced.apply(createPostJoinInferencePlan({"stream1$pixels", "stream2$pixels"}, std::nullopt))),
        PostJoinInferenceImplementation::BATCHED);
}

TEST_F(InferenceImplementationSelectorTest, IgnoresSingleFieldInference)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});
    const InferenceImplementationSelector selector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), profiler};

    const auto plan = selector.apply(createPostJoinInferencePlan({"stream1$pixels"}, worker));

    EXPECT_EQ(getSelectedImplementation(plan), std::nullopt);
    EXPECT_TRUE(profiler->jobs.empty());
}

}
}
