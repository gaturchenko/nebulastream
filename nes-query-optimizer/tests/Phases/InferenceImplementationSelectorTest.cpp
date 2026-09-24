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
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/FieldAccessLogicalFunction.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <Phases/InferenceImplementationSelector.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Traits/InferenceBatchSizeTrait.hpp>
#include <Traits/PlacementTrait.hpp>
#include <Traits/PostJoinInferenceImplementationTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <WindowTypes/Measures/TimeCharacteristic.hpp>
#include <WindowTypes/Measures/TimeMeasure.hpp>
#include <WindowTypes/Types/SlidingWindow.hpp>
#include <WindowTypes/Types/WindowType.hpp>
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

/// Answers profiling jobs with per-invocation latencies given as a function of the batch size and records the submitted jobs.
class FakeInferenceProfiler final : public InferenceProfiler
{
public:
    explicit FakeInferenceProfiler(std::function<std::chrono::nanoseconds(size_t)> latency) : latency(std::move(latency)) { }

    FakeInferenceProfiler(std::chrono::nanoseconds singleTupleLatency, std::chrono::nanoseconds batchedLatency)
        : latency([singleTupleLatency, batchedLatency](const size_t batchSize)
                  { return batchSize == 1 ? singleTupleLatency : batchedLatency; })
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
            const auto batchLatency = latency(batchSize);
            profile.batches.push_back(BatchProfile{
                .batchSize = batchSize,
                .iterations = request.measurementIterations,
                .medianLatency = batchLatency,
                .meanLatency = batchLatency,
                .minLatency = batchLatency});
        }
        return profile;
    }

    std::function<std::chrono::nanoseconds(size_t)> latency;
    bool failing = false;
    std::vector<std::pair<Host, std::vector<size_t>>> jobs;
};

/// Invocation latency growing linearly with the batch size.
std::function<std::chrono::nanoseconds(size_t)> linearLatency(std::chrono::microseconds base, std::chrono::microseconds perTuple)
{
    return [base, perTuple](const size_t batchSize) { return base + perTuple * static_cast<int64_t>(batchSize); };
}

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

    /// A placed single-field inference operator consuming the results of `window`, or of a source if no window is given.
    static LogicalPlan
    createWindowedInferencePlan(const std::optional<std::shared_ptr<Windowing::WindowType>>& window, const bool keyed = false)
    {
        auto inferModel = InferModelLogicalOperator(loadVarsizedModel(), {"pixels"});
        TraitSet traitSet;
        tryInsert(traitSet, PlacementTrait{worker});
        inferModel = inferModel.withTraitSet(traitSet);
        if (window.has_value())
        {
            std::vector<FieldAccessLogicalFunction> groupingKeys;
            if (keyed)
            {
                groupingKeys.emplace_back("id");
            }
            const auto windowedAggregation = WindowedAggregationLogicalOperator(std::move(groupingKeys), {}, window.value());
            inferModel
                = inferModel.withChildren({LogicalOperator{SequenceLogicalOperator(SequenceLogicalOperator::SequenceSource::INFERENCE, 256)
                                                               .withChildren({LogicalOperator{windowedAggregation}})}});
        }
        return LogicalPlan{INVALID_QUERY_ID, {LogicalOperator{inferModel}}};
    }

    static std::shared_ptr<Windowing::WindowType> createSlidingWindow(const uint64_t sizeMs, const uint64_t strideMs)
    {
        return Windowing::SlidingWindow::of(
            Windowing::TimeCharacteristic::createIngestionTime(), Windowing::TimeMeasure(sizeMs), Windowing::TimeMeasure(strideMs));
    }

    static QueryOptimizerConfiguration createBatchSizeConfiguration()
    {
        QueryOptimizerConfiguration configuration;
        configuration.inferenceBatchSizeStrategy.setValue(InferenceBatchSizeStrategy::OPTIMIZER_CHOOSES);
        return configuration;
    }

    static std::optional<uint64_t> getSelectedBatchSize(const LogicalPlan& plan)
    {
        const auto trait = getTrait<InferenceBatchSizeTrait>(plan.getRootOperators().front().getTraitSet());
        if (!trait.has_value())
        {
            return std::nullopt;
        }
        return trait.value()->batchSize;
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

/// Window of 200 ms, stride 10 ms: the fill time admits candidates up to 16. With 1 ms + 0.2 ms per tuple, batch 16 takes 4.2 ms
/// (hard SLO 10 ms) and its first result 154.2 ms (weak SLO 200 ms), and it has the highest throughput.
TEST_F(InferenceImplementationSelectorTest, SelectsHighestThroughputBatchSizeMeetingBothSlos)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));

    EXPECT_EQ(getSelectedBatchSize(plan), 16);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().first, worker);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2, 4, 8, 16}));
}

/// With 2 ms + 1 ms per tuple, batch 16 takes 18 ms and misses the 10 ms stride, batch 8 takes exactly 10 ms.
TEST_F(InferenceImplementationSelectorTest, HardSloBoundsTheInvocationLatency)
{
    auto profiler
        = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{2000}, std::chrono::microseconds{1000}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 8);
}

/// Window of 35 ms, stride 10 ms: only candidates up to 4 can fill in time. With a constant 6 ms invocation, the first result of
/// batch 4 takes 36 ms and misses the weak SLO, batch 2 takes 16 ms.
TEST_F(InferenceImplementationSelectorTest, WeakSloBoundsTheBatchFillTime)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>([](size_t) { return std::chrono::nanoseconds{std::chrono::milliseconds{6}}; });
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)));

    EXPECT_EQ(getSelectedBatchSize(plan), 2);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2, 4}));
}

/// Replaying twice as fast halves the wall-clock stride to 5 ms, so batch 4 (6 ms) misses the hard SLO that it met in real time.
TEST_F(InferenceImplementationSelectorTest, ReplaySpeedupScalesTheSlos)
{
    auto profiler
        = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{2000}, std::chrono::microseconds{1000}));
    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceReplaySpeedup.setValue(2.0F);
    const InferenceImplementationSelector selector{configuration, profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 2);
}

TEST_F(InferenceImplementationSelectorTest, RespectsMaximumBatchSize)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceMaxBatchSize.setValue(4);
    const InferenceImplementationSelector selector{configuration, profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 4);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2, 4}));
}

/// If even a single-tuple invocation misses the stride, the highest-throughput batch size is selected to keep up as well as possible.
TEST_F(InferenceImplementationSelectorTest, FallsBackToHighestThroughputIfNoBatchSizeMeetsTheHardSlo)
{
    auto profiler
        = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{20000}, std::chrono::microseconds{1000}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 16);
}

TEST_F(InferenceImplementationSelectorTest, ProfilesOnlyUncachedBatchSizes)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    (void)selector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)));
    (void)selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));

    ASSERT_EQ(profiler->jobs.size(), 2);
    EXPECT_EQ(profiler->jobs.at(0).second, (std::vector<size_t>{1, 2, 4}));
    EXPECT_EQ(profiler->jobs.at(1).second, (std::vector<size_t>{8, 16}));
}

TEST_F(InferenceImplementationSelectorTest, UsesConfiguredBatchSizeWithoutUnkeyedTimeWindow)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceBatchSize.setValue(8);
    const InferenceImplementationSelector selector{configuration, profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(std::nullopt))), 8);
    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10), true))), 8);
    EXPECT_TRUE(profiler->jobs.empty());
}

TEST_F(InferenceImplementationSelectorTest, UsesConfiguredBatchSizeIfProfilingFails)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    profiler->failing = true;
    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceBatchSize.setValue(4);
    const InferenceImplementationSelector selector{configuration, profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 4);
}

TEST_F(InferenceImplementationSelectorTest, FixedBatchSizeStrategyLeavesThePlanUnchanged)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    const InferenceImplementationSelector selector{QueryOptimizerConfiguration{}, profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), std::nullopt);
    EXPECT_TRUE(profiler->jobs.empty());
}

TEST_F(InferenceImplementationSelectorTest, RemovesInferenceSequenceForSingleTupleInference)
{
    const auto hasInferenceSequence = [](const LogicalPlan& plan)
    {
        const auto children = plan.getRootOperators().front().getChildren();
        return children.size() == 1 && children.front().tryGetAs<SequenceLogicalOperator>().has_value();
    };

    /// A constant 2 ms invocation makes batch 16 the fastest.
    auto batching = std::make_shared<FakeInferenceProfiler>([](size_t) { return std::chrono::nanoseconds{std::chrono::milliseconds{2}}; });
    const InferenceImplementationSelector batchingSelector{createBatchSizeConfiguration(), batching};
    const auto batchedPlan = batchingSelector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));
    EXPECT_EQ(getSelectedBatchSize(batchedPlan), 16);
    EXPECT_TRUE(hasInferenceSequence(batchedPlan));

    /// With 1 ms per tuple, all batch sizes have the same throughput and the smallest one wins.
    auto noBatching = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{0}, std::chrono::microseconds{1000}));
    const InferenceImplementationSelector singleTupleSelector{createBatchSizeConfiguration(), noBatching};
    const auto singleTuplePlan = singleTupleSelector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));
    EXPECT_EQ(getSelectedBatchSize(singleTuplePlan), 1);
    EXPECT_FALSE(hasInferenceSequence(singleTuplePlan));
    ASSERT_EQ(singleTuplePlan.getRootOperators().front().getChildren().size(), 1);
    EXPECT_TRUE(
        singleTuplePlan.getRootOperators().front().getChildren().front().tryGetAs<WindowedAggregationLogicalOperator>().has_value());
}

}
}
