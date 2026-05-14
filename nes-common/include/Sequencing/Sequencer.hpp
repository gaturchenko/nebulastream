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

#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include <Sequencing/SequenceData.hpp>
#include <folly/Synchronized.h>
#include <ErrorHandling.hpp>

namespace NES
{

/// Buffers can arrive out of order when multiple source tasks run concurrently.
/// Sequencer stores out-of-order items and releases them by sequence/chunk order.
template <typename T>
class Sequencer
{
public:
    std::optional<T> isNext(SequenceData sequence, T data)
    {
        auto state = stateMtx.ulock();
        if (sequence.sequenceNumber == state->nextSequence && sequence.chunkNumber == state->nextChunkNumber)
        {
            return data;
        }

        auto writableState = std::move(state).moveFromUpgradeToWrite();
        writableState->queue.emplace(sequence, std::move(data));
        return std::nullopt;
    }

    std::optional<T> advanceAndGetNext(SequenceData sequence)
    {
        auto state = stateMtx.wlock();
        PRECONDITION(
            state->nextChunkNumber == sequence.chunkNumber && state->nextSequence == sequence.sequenceNumber,
            "advance was called with invalid sequence. Expected: {} but received: {}",
            SequenceData(SequenceNumber(state->nextSequence), ChunkNumber(state->nextChunkNumber), false),
            sequence);

        if (sequence.lastChunk)
        {
            state->nextSequence = sequence.sequenceNumber + 1;
            state->nextChunkNumber = ChunkNumber::INITIAL;
        }
        else
        {
            state->nextChunkNumber = sequence.chunkNumber + 1;
        }

        if (!state->queue.empty() && state->queue.top().first.sequenceNumber == state->nextSequence
            && state->queue.top().first.chunkNumber == state->nextChunkNumber)
        {
            auto next = std::move(state->queue.top().second);
            state->queue.pop();
            return next;
        }
        return std::nullopt;
    }

    void reset()
    {
        auto state = stateMtx.wlock();
        *state = State{};
    }

private:
    struct CompareQueueElements
    {
        bool operator()(const std::pair<SequenceData, T>& lhs, const std::pair<SequenceData, T>& rhs) const
        {
            return rhs.first < lhs.first;
        }
    };

    struct State
    {
        SequenceNumber::Underlying nextSequence = SequenceNumber::INITIAL;
        ChunkNumber::Underlying nextChunkNumber = ChunkNumber::INITIAL;
        std::priority_queue<std::pair<SequenceData, T>, std::vector<std::pair<SequenceData, T>>, CompareQueueElements> queue;
    };

    folly::Synchronized<State> stateMtx;
};

}
