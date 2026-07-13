# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(SARAD_SIZE_200_STRIDE_10_UDF LogicalFunction nes-logical-operators src/ModelZooUDFLogicalFunction.cpp)
add_plugin(MHATT_RNN_UDF LogicalFunction nes-logical-operators)
add_plugin(SQUEEZEDET_UDF LogicalFunction nes-logical-operators)
add_plugin(SQUEEZEDET_192X624_UDF LogicalFunction nes-logical-operators)

add_plugin_as_library(
        SARAD_SIZE_200_STRIDE_10_UDF
        PhysicalFunction
        nes-physical-operators
        sarad-size-200-stride-10-udf
        src/SaradSize200Stride10UDFPhysicalFunction.cpp)

add_plugin_as_library(
        MHATT_RNN_UDF
        PhysicalFunction
        nes-physical-operators
        mhatt-rnn-udf
        src/MhattRnnUDFPhysicalFunction.cpp)

add_plugin_as_library(
        SQUEEZEDET_UDF
        PhysicalFunction
        nes-physical-operators
        squeezedet-udf
        src/SqueezeDetUDFPhysicalFunction.cpp)

add_plugin_as_library(
        SQUEEZEDET_192X624_UDF
        PhysicalFunction
        nes-physical-operators
        squeezedet-192x624-udf
        src/SqueezeDet192x624UDFPhysicalFunction.cpp)

foreach (model_zoo_udf_target
        sarad-size-200-stride-10-udf
        mhatt-rnn-udf
        squeezedet-udf
        squeezedet-192x624-udf)
    target_include_directories(${model_zoo_udf_target} PRIVATE
            ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
            ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)

    target_compile_definitions(${model_zoo_udf_target} PRIVATE
            NES_MODEL_ZOO_UDF_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            NES_MODEL_ZOO_UDF_BINARY_DIR="${PROJECT_BINARY_DIR}")

    target_link_libraries(${model_zoo_udf_target} PRIVATE nes-inference-runtime)
endforeach ()
