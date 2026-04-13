# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


add_plugin(ToBase64 LogicalFunction nes-logical-operators ToBase64LogicalFunction.cpp)
add_plugin_as_library(ToBase64 PhysicalFunction nes-physical-operators to-base-64 ToBase64PhysicalFunction.cpp)

add_plugin(FromBase64 LogicalFunction nes-logical-operators FromBase64LogicalFunction.cpp)
add_plugin_as_library(FromBase64 PhysicalFunction nes-physical-operators from-base-64 FromBase64PhysicalFunction.cpp)

find_package(OpenSSL CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

target_link_libraries(from-base-64 PRIVATE OpenSSL::SSL)
target_link_libraries(to-base-64 PRIVATE OpenSSL::SSL)
