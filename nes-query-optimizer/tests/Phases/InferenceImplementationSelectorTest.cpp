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
#include <Traits/PredictionCacheScopeTrait.hpp>
#include <Traits/SharedCompiledModelTrait.hpp>
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
#include <InferenceConfiguration.hpp>
#include <InferenceProfiler.hpp>
#include <ModelCatalog.hpp>
#include <ModelProfile.hpp>
#include <QueryId.hpp>
#include <QueryOptimizerConfiguration.hpp>

namespace NES
{
namespace
{

/// A model whose sessions cost 8 MB for the first and 6 MB for each further private one, or 1 MB each when they share the
/// compiled model, on a worker with 4 threads and 8 GB of memory.
InferenceMemoryProfile defaultMemoryProfile()
{
    return InferenceMemoryProfile{
        .session = SessionMemoryProfile{.measuredBatchSize = 1, .modelPayloadBytes = 4096, .inputBufferBytes = 3136, .outputBufferBytes = 40, .firstPrivateSessionBytes = 8 * 1024 * 1024, .additionalPrivateSessionBytes = 6 * 1024 * 1024, .firstSharedSessionBytes = 8 * 1024 * 1024, .additionalSharedSessionBytes = 1024 * 1024, .measured = true},
        .predictionCache = PredictionCacheMemoryProfile{},
        .worker = WorkerMemoryProfile{.physicalMemoryBytes = 8UL * 1024 * 1024 * 1024, .bufferPoolBytes = 0, .numberOfWorkerThreads = 4}};
}

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
        if (request.assessMemoryForBatchSize.has_value())
        {
            memoryJobs.emplace_back(worker, request.assessMemoryForBatchSize.value());
        }
        else
        {
            jobs.emplace_back(worker, request.batchSizes);
        }
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
        if (request.assessMemoryForBatchSize.has_value())
        {
            profile.memory = memory;
            profile.memory.session.measuredBatchSize = request.assessMemoryForBatchSize.value();
        }
        return profile;
    }

    std::function<std::chrono::nanoseconds(size_t)> latency;
    std::vector<std::pair<Host, size_t>> memoryJobs;
    InferenceMemoryProfile memory = defaultMemoryProfile();
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
/// of the 160 ms in which its batch arrives (utilization 0.03) and its first result 154.2 ms (SLO 0.9 * 200 ms), and it has the highest
/// throughput.
TEST_F(InferenceImplementationSelectorTest, SelectsHighestThroughputBatchSizeThatKeepsUpAndMeetsTheSlo)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));

    EXPECT_EQ(getSelectedBatchSize(plan), 16);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().first, worker);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2, 4, 8, 16}));
}

/// With 2 ms + 1 ms per tuple, batch 16 takes 18 ms, longer than the 10 ms stride, but a batch of 16 arrives only every 160 ms, so
/// it keeps up (utilization 0.11) and its first result takes 168 ms (SLO 180 ms).
TEST_F(InferenceImplementationSelectorTest, AdmitsBatchInvocationsLongerThanTheStride)
{
    auto profiler
        = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{2000}, std::chrono::microseconds{1000}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 16);
}

/// Window of 35 ms, stride 10 ms: only candidates up to 4 can fill in time. With a constant 6 ms invocation, the first result of
/// batch 4 takes 36 ms and misses the SLO of 31.5 ms, batch 2 takes 16 ms.
TEST_F(InferenceImplementationSelectorTest, LatencySloBoundsTheBatchFillTime)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>([](size_t) { return std::chrono::nanoseconds{std::chrono::milliseconds{6}}; });
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)));

    EXPECT_EQ(getSelectedBatchSize(plan), 2);
    ASSERT_EQ(profiler->jobs.size(), 1);
    EXPECT_EQ(profiler->jobs.front().second, (std::vector<size_t>{1, 2, 4}));
}

/// With 5 ms + 0.5 ms per tuple, the first result of batch 16 takes 163 ms in real time (SLO 180 ms). Replaying four times as fast
/// shrinks the wall-clock window to 50 ms (SLO 45 ms) and the stride to 2.5 ms, so batch 16 takes 50.5 ms and misses the SLO, batch
/// 8 26.5 ms.
TEST_F(InferenceImplementationSelectorTest, ReplaySpeedupScalesTheSlo)
{
    const auto latency = linearLatency(std::chrono::microseconds{5000}, std::chrono::microseconds{500});
    const InferenceImplementationSelector realTimeSelector{createBatchSizeConfiguration(), std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(realTimeSelector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 16);

    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceReplaySpeedup.setValue(4.0F);
    const InferenceImplementationSelector selector{configuration, std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 8);
}

/// With 1 ms + 0.2 ms per tuple, the first result of batch 16 takes 154.2 ms, within the default SLO of 0.9 * 200 ms. Leaving
/// three quarters of the window to the predicted latency (150 ms) excludes it, and batch 8 (72.6 ms) is selected.
TEST_F(InferenceImplementationSelectorTest, LatencySloFractionLeavesRoomForUnprofiledDelays)
{
    const auto latency = linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200});
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 16);

    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceLatencySloFraction.setValue(0.75F);
    const InferenceImplementationSelector strictSelector{configuration, std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(strictSelector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)))), 8);
}

/// Window of 35 ms, stride 10 ms: candidates up to 4. Batch 1 (9.5 ms, utilization 0.95) and batch 2 (17 ms, 0.85) exceed the
/// default maximum utilization of 0.8; only batch 4 (20 ms, 0.5) keeps up, although its first result takes 50 ms and misses the 31.5 ms SLO.
/// Falling behind would let the latency grow without bound, so batch 4 is selected. Allowing a utilization of 0.9 admits batch 2,
/// which keeps up and meets the SLO (27 ms).
TEST_F(InferenceImplementationSelectorTest, PrefersKeepingUpOverTheLatencySlo)
{
    const auto latency = [](const size_t batchSize) -> std::chrono::nanoseconds
    {
        switch (batchSize)
        {
            case 1:
                return std::chrono::microseconds{9500};
            case 2:
                return std::chrono::microseconds{17000};
            default:
                return std::chrono::microseconds{20000};
        }
    };
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(selector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)))), 4);

    auto configuration = createBatchSizeConfiguration();
    configuration.inferenceMaxUtilization.setValue(0.9F);
    const InferenceImplementationSelector relaxedSelector{configuration, std::make_shared<FakeInferenceProfiler>(latency)};
    EXPECT_EQ(getSelectedBatchSize(relaxedSelector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)))), 2);
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

/// With 200 ms + 1 ms per tuple, even batch 16 (216 ms every 160 ms) falls behind, so the highest-throughput batch size is selected
/// to fall behind as little as possible.
TEST_F(InferenceImplementationSelectorTest, FallsBackToHighestThroughputIfNoBatchSizeKeepsUp)
{
    auto profiler
        = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{200000}, std::chrono::microseconds{1000}));
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

/// The memory assessment costs extra runtime sessions on the worker, so it must happen once per model and worker, and the sessions
/// must be set up for the batch size the operator was decided to use.
TEST_F(InferenceImplementationSelectorTest, AssessesMemoryOncePerModelAndWorkerForTheSelectedBatchSize)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));
    ASSERT_EQ(getSelectedBatchSize(plan), 16);
    ASSERT_EQ(profiler->memoryJobs.size(), 1);
    EXPECT_EQ(profiler->memoryJobs.front().first, worker);
    EXPECT_EQ(profiler->memoryJobs.front().second, 16);

    const auto memory = selector.getMemoryProfile(worker, loadVarsizedModel());
    ASSERT_TRUE(memory.has_value());
    EXPECT_TRUE(memory->session.measured);
    EXPECT_EQ(memory->session.measuredBatchSize, 16);
    EXPECT_EQ(memory->session.privateDeploymentBytes(4), (8 * 1024 * 1024) + (3 * 6 * 1024 * 1024));
    EXPECT_EQ(memory->session.sharedDeploymentBytes(4), (8 * 1024 * 1024) + (3 * 1024 * 1024));

    /// A second query still profiles latencies it has not measured yet, but does not assess the memory again.
    (void)selector.apply(createWindowedInferencePlan(createSlidingWindow(35, 10)));
    EXPECT_EQ(profiler->memoryJobs.size(), 1);
    EXPECT_FALSE(selector.getMemoryProfile(Host{"worker-2:8080"}, loadVarsizedModel()).has_value());
}

/// Post-join inference invokes the model with one batch of k payloads or with single tuples, and the assessment follows that choice.
TEST_F(InferenceImplementationSelectorTest, AssessesMemoryForThePostJoinInvocation)
{
    auto batching = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{120});
    const InferenceImplementationSelector batchedSelector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), batching};
    (void)batchedSelector.apply(createPostJoinInferencePlan({"s1$pixels", "s2$pixels", "s3$pixels"}, worker));
    ASSERT_EQ(batching->memoryJobs.size(), 1);
    EXPECT_EQ(batching->memoryJobs.front().second, 3);

    auto sequential = std::make_shared<FakeInferenceProfiler>(std::chrono::nanoseconds{100}, std::chrono::nanoseconds{195});
    const InferenceImplementationSelector sequentialSelector{createConfiguration(PostJoinInferenceStrategy::OPTIMIZER_CHOOSES), sequential};
    (void)sequentialSelector.apply(createPostJoinInferencePlan({"s1$pixels", "s2$pixels"}, worker));
    ASSERT_EQ(sequential->memoryJobs.size(), 1);
    EXPECT_EQ(sequential->memoryJobs.front().second, 1);
}

TEST_F(InferenceImplementationSelectorTest, DoesNotAssessMemoryWithoutADecisionToMake)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{200}));

    const InferenceImplementationSelector fixedBatchSize{QueryOptimizerConfiguration{}, profiler};
    (void)fixedBatchSize.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));

    const InferenceImplementationSelector fixedPostJoin{createConfiguration(PostJoinInferenceStrategy::BATCHED), profiler};
    (void)fixedPostJoin.apply(createPostJoinInferencePlan({"s1$pixels", "s2$pixels"}, worker));

    EXPECT_TRUE(profiler->memoryJobs.empty());
}

/// Deployment decisions: the memory-saving alternatives are only taken when the faster ones do not fit the worker's budget.
class InferenceDeploymentTest : public InferenceImplementationSelectorTest
{
public:
    static QueryOptimizerConfiguration createDeploymentConfiguration()
    {
        auto configuration = createBatchSizeConfiguration();
        configuration.inferenceMemoryStrategy.setValue(InferenceMemoryStrategy::OPTIMIZER_CHOOSES);
        return configuration;
    }

    /// A plan whose inference operator invokes the model with single tuples, which is what a global prediction cache requires.
    static LogicalPlan createSingleTupleInferencePlan()
    {
        /// One invocation per tuple takes 1 ms at every batch size, so the smallest batch wins.
        return createWindowedInferencePlan(createSlidingWindow(200, 10));
    }

    static std::shared_ptr<FakeInferenceProfiler> createProfiler()
    {
        return std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{0}, std::chrono::microseconds{1000}));
    }

    static std::optional<bool> getSharedCompiledModel(const LogicalPlan& plan)
    {
        const auto trait = getTrait<SharedCompiledModelTrait>(plan.getRootOperators().front().getTraitSet());
        if (!trait.has_value())
        {
            return std::nullopt;
        }
        return trait.value()->shareCompiledModel;
    }

    static std::optional<PredictionCacheScope> getPredictionCacheScope(const LogicalPlan& plan)
    {
        const auto trait = getTrait<PredictionCacheScopeTrait>(plan.getRootOperators().front().getTraitSet());
        if (!trait.has_value())
        {
            return std::nullopt;
        }
        return trait.value()->scope;
    }
};

TEST_F(InferenceDeploymentTest, KeepsPrivateCompiledModelsIfTheyFitTheBudget)
{
    auto profiler = createProfiler();
    const InferenceImplementationSelector selector{createDeploymentConfiguration(), profiler};

    const auto plan = selector.apply(createSingleTupleInferencePlan());

    EXPECT_EQ(getSharedCompiledModel(plan), false);
}

/// 4 sessions need 8 + 3 x 6 = 26 MB privately and 8 + 3 x 1 = 11 MB sharing one compiled model. A budget between the two forces
/// sharing.
TEST_F(InferenceDeploymentTest, SharesTheCompiledModelIfPrivateCopiesDoNotFit)
{
    auto profiler = createProfiler();
    profiler->memory.worker.physicalMemoryBytes = 20 * 1024 * 1024;
    auto configuration = createDeploymentConfiguration();
    configuration.inferenceMemoryBudgetFraction.setValue(1.0F);
    const InferenceImplementationSelector selector{configuration, profiler};

    const auto plan = selector.apply(createSingleTupleInferencePlan());

    EXPECT_EQ(getSharedCompiledModel(plan), true);
}

TEST_F(InferenceDeploymentTest, KeepsThreadLocalCachesIfTheyFitTheBudget)
{
    auto profiler = createProfiler();
    profiler->memory.predictionCache = PredictionCacheMemoryProfile{
        .entries = 1024, .bytesPerEntry = 1024, .threadLocalBytes = 4 * 1024 * 1024, .globalBytes = 1024 * 1024};
    const InferenceImplementationSelector selector{createDeploymentConfiguration(), profiler};

    const auto plan = selector.apply(createSingleTupleInferencePlan());

    EXPECT_EQ(getPredictionCacheScope(plan), PredictionCacheScope::THREAD_LOCAL);
    EXPECT_EQ(getSharedCompiledModel(plan), false);
}

/// Sessions cost 26 MB privately or 11 MB shared, caches 40 MB thread-local or 10 MB global. Only sharing both fits into 55 MB,
/// and the cache is given up last because its lock is on the path of every record.
TEST_F(InferenceDeploymentTest, FallsBackToAGlobalCacheOnlyAfterSharingTheCompiledModel)
{
    auto profiler = createProfiler();
    profiler->memory.predictionCache = PredictionCacheMemoryProfile{
        .entries = 1024, .bytesPerEntry = 40960, .threadLocalBytes = 40 * 1024 * 1024, .globalBytes = 10 * 1024 * 1024};
    auto configuration = createDeploymentConfiguration();
    configuration.inferenceMemoryBudgetFraction.setValue(1.0F);

    /// 11 + 40 = 51 MB fits: sharing the compiled model alone is enough.
    profiler->memory.worker.physicalMemoryBytes = 55 * 1024 * 1024;
    const InferenceImplementationSelector sharingSelector{configuration, profiler};
    const auto sharingPlan = sharingSelector.apply(createSingleTupleInferencePlan());
    EXPECT_EQ(getSharedCompiledModel(sharingPlan), true);
    EXPECT_EQ(getPredictionCacheScope(sharingPlan), PredictionCacheScope::THREAD_LOCAL);

    /// 11 + 10 = 21 MB is the only combination that fits into 25 MB.
    auto tightProfiler = createProfiler();
    tightProfiler->memory.predictionCache = profiler->memory.predictionCache;
    tightProfiler->memory.worker.physicalMemoryBytes = 25 * 1024 * 1024;
    const InferenceImplementationSelector tightSelector{configuration, tightProfiler};
    const auto tightPlan = tightSelector.apply(createSingleTupleInferencePlan());
    EXPECT_EQ(getSharedCompiledModel(tightPlan), true);
    EXPECT_EQ(getPredictionCacheScope(tightPlan), PredictionCacheScope::GLOBAL);
}

/// Batched inference cannot use a global cache, so only the compiled model can be shared however tight the budget is.
TEST_F(InferenceDeploymentTest, KeepsThreadLocalCachesForBatchedInference)
{
    auto profiler = std::make_shared<FakeInferenceProfiler>(linearLatency(std::chrono::microseconds{1000}, std::chrono::microseconds{10}));
    profiler->memory.predictionCache = PredictionCacheMemoryProfile{
        .entries = 1024, .bytesPerEntry = 40960, .threadLocalBytes = 40 * 1024 * 1024, .globalBytes = 10 * 1024 * 1024};
    profiler->memory.worker.physicalMemoryBytes = 1024 * 1024;
    auto configuration = createDeploymentConfiguration();
    configuration.inferenceMemoryBudgetFraction.setValue(1.0F);
    const InferenceImplementationSelector selector{configuration, profiler};

    const auto plan = selector.apply(createWindowedInferencePlan(createSlidingWindow(200, 10)));

    ASSERT_GT(getSelectedBatchSize(plan), 1);
    EXPECT_EQ(getPredictionCacheScope(plan), PredictionCacheScope::THREAD_LOCAL);
    EXPECT_EQ(getSharedCompiledModel(plan), true);
}

TEST_F(InferenceDeploymentTest, LeavesTheDeploymentToTheWorkerUnderTheFixedStrategy)
{
    auto profiler = createProfiler();
    const InferenceImplementationSelector selector{createBatchSizeConfiguration(), profiler};

    const auto plan = selector.apply(createSingleTupleInferencePlan());

    EXPECT_EQ(getSharedCompiledModel(plan), std::nullopt);
    EXPECT_EQ(getPredictionCacheScope(plan), std::nullopt);
}

}
}
