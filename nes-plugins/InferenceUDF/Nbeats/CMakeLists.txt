# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(NBEATS_SIZE_120_STRIDE_12_UDF LogicalFunction nes-logical-operators src/NbeatsUDFLogicalFunction.cpp)
add_plugin(NBEATS_SIZE_300_STRIDE_30_UDF LogicalFunction nes-logical-operators)
add_plugin(NBEATS_SIZE_600_STRIDE_60_UDF LogicalFunction nes-logical-operators)

add_plugin_as_library(
        NBEATS_SIZE_120_STRIDE_12_UDF
        PhysicalFunction
        nes-physical-operators
        nbeats-size-120-stride-12-udf
        src/NbeatsSize120Stride12UDFPhysicalFunction.cpp)

add_plugin_as_library(
        NBEATS_SIZE_300_STRIDE_30_UDF
        PhysicalFunction
        nes-physical-operators
        nbeats-size-300-stride-30-udf
        src/NbeatsSize300Stride30UDFPhysicalFunction.cpp)

add_plugin_as_library(
        NBEATS_SIZE_600_STRIDE_60_UDF
        PhysicalFunction
        nes-physical-operators
        nbeats-size-600-stride-60-udf
        src/NbeatsSize600Stride60UDFPhysicalFunction.cpp)

foreach (nbeats_udf_target
        nbeats-size-120-stride-12-udf
        nbeats-size-300-stride-30-udf
        nbeats-size-600-stride-60-udf)
    target_include_directories(${nbeats_udf_target} PRIVATE
            ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
            ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)

    target_compile_definitions(${nbeats_udf_target} PRIVATE
            NES_NBEATS_UDF_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            NES_NBEATS_UDF_BINARY_DIR="${PROJECT_BINARY_DIR}")

    target_link_libraries(${nbeats_udf_target} PRIVATE nes-inference-runtime)
endforeach ()
