# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(CURRENT_TIME LogicalFunction nes-logical-operators src/CurrentTimeLogicalFunction.cpp)
add_plugin_as_library(CURRENT_TIME PhysicalFunction nes-physical-operators current-time src/CurrentTimePhysicalFunction.cpp)

target_include_directories(current-time PRIVATE
        ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
        ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)
