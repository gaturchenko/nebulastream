# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin(RESIZE_IMAGE LogicalFunction nes-logical-operators src/ResizeImageLogicalFunction.cpp)
add_plugin_as_library(RESIZE_IMAGE PhysicalFunction nes-physical-operators resize-image src/ResizeImagePhysicalFunction.cpp)

add_plugin(CONVERT_COLOR_SCHEME LogicalFunction nes-logical-operators src/ConvertColorSchemeLogicalFunction.cpp)
add_plugin_as_library(
        CONVERT_COLOR_SCHEME
        PhysicalFunction
        nes-physical-operators
        convert-color-scheme
        src/ConvertColorSchemePhysicalFunction.cpp)

add_plugin(SSIM LogicalFunction nes-logical-operators src/SSIMLogicalFunction.cpp)
add_plugin_as_library(SSIM PhysicalFunction nes-physical-operators ssim src/SSIMPhysicalFunction.cpp)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

foreach(image_manip_physical_plugin resize-image convert-color-scheme ssim)
    target_include_directories(${image_manip_physical_plugin} PRIVATE
            ${PROJECT_SOURCE_DIR}/nes-physical-operators/registry/include
            ${PROJECT_BINARY_DIR}/nes-physical-operators/registry/templates)
    target_link_libraries(${image_manip_physical_plugin} PRIVATE ${OpenCV_LIBS})
endforeach()
