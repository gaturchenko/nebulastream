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

#include <cstdint>

namespace NES::AudioDenoising
{

enum class SpectralDenoisingMode : uint8_t
{
    Subtraction,
    HardGate
};

uint64_t denoiseRawFloat32(
    int8_t* inputData, uint64_t inputSize, double noiseRmsDbfs, SpectralDenoisingMode mode, int8_t* outputData, uint64_t outputCapacity);

}
