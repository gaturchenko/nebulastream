# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


add_plugin(ToBase64 LogicalFunction nes-logical-operators src/ToBase64LogicalFunction.cpp)
add_plugin_as_library(ToBase64 PhysicalFunction nes-physical-operators to-base-64 src/ToBase64PhysicalFunction.cpp)

add_plugin(FromBase64 LogicalFunction nes-logical-operators src/FromBase64LogicalFunction.cpp)
add_plugin_as_library(FromBase64 PhysicalFunction nes-physical-operators from-base-64 src/FromBase64PhysicalFunction.cpp)

add_plugin(SSIM LogicalFunction nes-logical-operators src/SSIMLogicalFunction.cpp)
add_plugin_as_library(SSIM PhysicalFunction nes-physical-operators ssim-function src/SSIMPhysicalFunction.cpp)

add_plugin(ResizeImage LogicalFunction nes-logical-operators src/ResizeImageLogicalFunction.cpp)
add_plugin_as_library(ResizeImage PhysicalFunction nes-physical-operators resize-image src/ResizeImagePhysicalFunction.cpp)

add_plugin(ConvertColorScheme LogicalFunction nes-logical-operators src/ConvertColorSchemeLogicalFunction.cpp)
add_plugin_as_library(ConvertColorScheme PhysicalFunction nes-physical-operators convert-color-scheme src/ConvertColorSchemePhysicalFunction.cpp)

find_package(OpenSSL CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

target_link_libraries(from-base-64 PRIVATE OpenSSL::SSL)
target_link_libraries(to-base-64 PRIVATE OpenSSL::SSL)
target_link_libraries(ssim-function PRIVATE ${OpenCV_LIBS})
target_link_libraries(resize-image PRIVATE ${OpenCV_LIBS})
target_link_libraries(convert-color-scheme PRIVATE ${OpenCV_LIBS})
