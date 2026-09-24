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

#include <cstddef>

#include <Configuration/WorkerConfiguration.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <InferenceConfiguration.hpp>
#include <InferenceProfiling.hpp>

namespace NES
{
namespace
{

/// A CWRU-sized model: 9600 float inputs, one float output.
constexpr size_t INPUT_BYTES = 9600 * sizeof(float);
constexpr size_t OUTPUT_BYTES = sizeof(float);
constexpr size_t ENTRIES = 1024;

class InferenceProfilingTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("InferenceProfilingTest.log", LogLevel::LOG_DEBUG); }
};

TEST_F(InferenceProfilingTest, PredictionCacheMemoryIsZeroWithoutACache)
{
    EXPECT_EQ(estimatePredictionCacheMemory(PredictionCacheType::NONE, ENTRIES, INPUT_BYTES, OUTPUT_BYTES, 4).threadLocalBytes, 0);
    EXPECT_EQ(estimatePredictionCacheMemory(PredictionCacheType::LRU, 0, INPUT_BYTES, OUTPUT_BYTES, 4).threadLocalBytes, 0);
}

TEST_F(InferenceProfilingTest, PredictionCacheMemoryCoversTheCachedRecordsAndPredictions)
{
    const auto memory = estimatePredictionCacheMemory(PredictionCacheType::LRU, ENTRIES, INPUT_BYTES, OUTPUT_BYTES, 1);

    EXPECT_EQ(memory.entries, ENTRIES);
    /// The cached record and prediction dominate the per-entry cost, the bookkeeping adds a small constant.
    EXPECT_GT(memory.bytesPerEntry, INPUT_BYTES + OUTPUT_BYTES);
    EXPECT_LT(memory.bytesPerEntry, INPUT_BYTES + OUTPUT_BYTES + 256);
    EXPECT_GE(memory.threadLocalBytes, ENTRIES * (INPUT_BYTES + OUTPUT_BYTES));
    EXPECT_GE(memory.globalBytes, ENTRIES * (INPUT_BYTES + OUTPUT_BYTES));
}

/// The thread-local cache is replicated per worker thread, the global one is not: that difference is what the scope decision needs.
TEST_F(InferenceProfilingTest, ThreadLocalCacheScalesWithWorkerThreadsAndGlobalDoesNot)
{
    const auto singleThreaded = estimatePredictionCacheMemory(PredictionCacheType::LRU, ENTRIES, INPUT_BYTES, OUTPUT_BYTES, 1);
    const auto eightThreads = estimatePredictionCacheMemory(PredictionCacheType::LRU, ENTRIES, INPUT_BYTES, OUTPUT_BYTES, 8);

    EXPECT_EQ(eightThreads.threadLocalBytes, 8 * singleThreaded.threadLocalBytes);
    /// The global cache only grows by the per-thread scratch buffer.
    EXPECT_EQ(eightThreads.globalBytes - singleThreaded.globalBytes, 7 * OUTPUT_BYTES);
    EXPECT_LT(eightThreads.globalBytes, eightThreads.threadLocalBytes);
}

TEST_F(InferenceProfilingTest, WorkerMemoryReportsThePhysicalMemoryAndTheBufferPoolReservation)
{
    WorkerConfiguration configuration;
    configuration.defaultQueryExecution.operatorBufferSize.setValue(131072);
    configuration.numberOfBuffersInGlobalBufferManager.setValue(1024);
    configuration.queryEngine.numberOfWorkerThreads.setValue(4);

    const auto memory = getWorkerMemoryProfile(configuration);

    EXPECT_GT(memory.physicalMemoryBytes, 0);
    EXPECT_EQ(memory.bufferPoolBytes, 131072UL * 1024UL);
    EXPECT_EQ(memory.numberOfWorkerThreads, 4);
    EXPECT_EQ(memory.availableBytes(), memory.physicalMemoryBytes - memory.bufferPoolBytes);
}

/// A buffer pool larger than the machine's memory leaves nothing for inference rather than underflowing.
TEST_F(InferenceProfilingTest, AvailableMemoryIsZeroIfTheBufferPoolExceedsThePhysicalMemory)
{
    WorkerConfiguration configuration;
    configuration.defaultQueryExecution.operatorBufferSize.setValue(131072);
    configuration.numberOfBuffersInGlobalBufferManager.setValue(524288);

    const auto memory = getWorkerMemoryProfile(configuration);

    EXPECT_GT(memory.bufferPoolBytes, memory.physicalMemoryBytes);
    EXPECT_EQ(memory.availableBytes(), 0);
}

}
}
