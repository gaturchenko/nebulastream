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

#include "../include/SpectralGatingPhysicalFunction.hpp"

#include "../include/AudioDenoising.hpp"

#include <utility>

#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
uint64_t denoiseAudio(int8_t* inputData, uint64_t inputSize, double noiseRmsDbfs, int8_t* outputData, uint64_t outputCapacity)
{
    return AudioDenoising::denoiseRawFloat32(
        inputData, inputSize, noiseRmsDbfs, AudioDenoising::SpectralDenoisingMode::HardGate, outputData, outputCapacity);
}
}

SpectralGatingPhysicalFunction::SpectralGatingPhysicalFunction(
    PhysicalFunction audioPhysicalFunction, PhysicalFunction noiseDbfsPhysicalFunction)
    : audioPhysicalFunction(std::move(audioPhysicalFunction)), noiseDbfsPhysicalFunction(std::move(noiseDbfsPhysicalFunction))
{
}

VarVal SpectralGatingPhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto inputValue = audioPhysicalFunction.execute(record, arena);
    const auto inputAudio = inputValue.getRawValueAs<VariableSizedData>();
    const auto inputAudioSize = inputAudio.getSize();
    if (inputAudioSize == 0U)
    {
        return inputValue;
    }
    const auto noiseDbfs
        = noiseDbfsPhysicalFunction.execute(record, arena).castToType(DataType::Type::FLOAT64).getRawValueAs<nautilus::val<double>>();

    auto outputAudio = arena.allocateVariableSizedData(inputAudioSize);
    const auto writtenSize
        = nautilus::invoke(denoiseAudio, inputAudio.getContent(), inputAudioSize, noiseDbfs, outputAudio.getContent(), inputAudioSize);
    if (writtenSize == 0U)
    {
        return inputValue;
    }

    return VariableSizedData(outputAudio.getContent(), writtenSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterSPECTRAL_GATINGPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 2, "SPECTRAL_GATING function must have exactly two child functions");
    PRECONDITION(arguments.inputTypes.size() == 2, "SPECTRAL_GATING function expects exactly two input type descriptors");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "SPECTRAL_GATING first argument must be VARSIZED, but got {}",
        arguments.inputTypes[0]);
    PRECONDITION(
        arguments.inputTypes[1].isFloat(), "SPECTRAL_GATING second argument must be FLOAT32/FLOAT64, but got {}", arguments.inputTypes[1]);

    return SpectralGatingPhysicalFunction(arguments.childFunctions[0], arguments.childFunctions[1]);
}

}
