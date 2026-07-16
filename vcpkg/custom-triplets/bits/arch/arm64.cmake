# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Pin dependencies to the ARMv8.0-A baseline. The build host (Ampere Altra,
# ARMv8.2-A) otherwise lets the compiler inline ARMv8.1 LSE atomics (ldaddal
# etc.) into generic framework code -- notably OpenVINO's ARM Compute Library
# (arm_gemm) -- which SIGILLs on ARMv8.0 Raspberry Pis (Cortex-A53/A72).
# armv8-a makes the compiler emit outline/LL-SC atomics that run everywhere;
# ACL's specialized SIMD kernels append their own higher -march per file and
# stay runtime-dispatched, so this only floors the un-guarded baseline code.
set(VCPKG_C_FLAGS   "-march=armv8-a")
set(VCPKG_CXX_FLAGS "-march=armv8-a")

# boost-context and openvino do not recognize arm64
if (PORT STREQUAL "boost-context")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_SYSTEM_PROCESSOR=aarch64)
elseif (PORT STREQUAL "openvino")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DOV_CPU_ARM_TARGET_ARCH=arm64-v8a)
endif ()
