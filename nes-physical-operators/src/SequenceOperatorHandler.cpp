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

#include <SequenceOperatorHandler.hpp>

#include <memory>
#include <optional>
#include <utility>

#include <Runtime/TupleBuffer.hpp>
#include <Sequencing/SequenceData.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

std::optional<TupleBuffer*> SequenceOperatorHandler::getNextBuffer(TupleBuffer* tupleBuffer)
{
    PRECONDITION(tupleBuffer != nullptr, "tupleBuffer must not be null");
    const auto sequenceData = SequenceData(tupleBuffer->getSequenceNumber(), tupleBuffer->getChunkNumber(), tupleBuffer->isLastChunk());
    if (auto nextBuffer = sequencer.isNext(sequenceData, *tupleBuffer))
    {
        currentBuffer = std::move(*nextBuffer);
        return std::addressof(currentBuffer);
    }
    return std::nullopt;
}

std::optional<TupleBuffer*> SequenceOperatorHandler::markBufferAsDone(TupleBuffer* tupleBuffer)
{
    PRECONDITION(tupleBuffer == std::addressof(currentBuffer), "Sequence handler can only finish the current buffer");
    const auto sequenceData = SequenceData(tupleBuffer->getSequenceNumber(), tupleBuffer->getChunkNumber(), tupleBuffer->isLastChunk());
    if (auto nextBuffer = sequencer.advanceAndGetNext(sequenceData))
    {
        currentBuffer = std::move(*nextBuffer);
        return std::addressof(currentBuffer);
    }
    return std::nullopt;
}

}
