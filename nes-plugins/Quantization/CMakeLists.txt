# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(QUANTIZE LogicalFunction nes-logical-operators src/QuantizeLogicalFunction.cpp)
add_plugin_as_library(QUANTIZE PhysicalFunction nes-physical-operators quantize src/QuantizePhysicalFunction.cpp)

find_package(OpenVINO CONFIG REQUIRED COMPONENTS Runtime)

target_include_directories(quantize PRIVATE
        ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
        ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)

target_link_libraries(quantize PRIVATE openvino::runtime)
