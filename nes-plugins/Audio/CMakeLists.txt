# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(SPECTRAL_SUBTRACTION LogicalFunction nes-logical-operators src/SpectralSubtractionLogicalFunction.cpp)
add_plugin_as_library(
        SPECTRAL_SUBTRACTION
        PhysicalFunction
        nes-physical-operators
        spectral-subtraction
        src/SpectralSubtractionPhysicalFunction.cpp)

add_plugin(SPECTRAL_GATING LogicalFunction nes-logical-operators src/SpectralGatingLogicalFunction.cpp)
add_plugin_as_library(
        SPECTRAL_GATING
        PhysicalFunction
        nes-physical-operators
        spectral-gating
        src/SpectralGatingPhysicalFunction.cpp)

find_package(FFTW3 CONFIG REQUIRED)

add_library(audio-denoising STATIC src/AudioDenoising.cpp)
target_link_libraries(audio-denoising PRIVATE FFTW3::fftw3)

foreach(audio_physical_plugin spectral-subtraction spectral-gating)
    target_include_directories(${audio_physical_plugin} PRIVATE
            ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
            ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)
    target_link_libraries(${audio_physical_plugin} PRIVATE audio-denoising)
endforeach()
