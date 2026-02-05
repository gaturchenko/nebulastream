# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin_as_library(Sequence RewriteRule nes-query-optimizer sequence_operator_library
        LowerToPhysicalSequence.cpp
        SequencePhysicalOperator.cpp
        SequenceOperatorHandler.cpp
        InterBufferBatchingOperator.cpp
        InterBufferBatchingOperatorHandler.cpp)

target_include_directories(sequence_operator_library PRIVATE ../include)
target_link_libraries(sequence_operator_library PRIVATE
        nes-physical-operators
        nes-query-optimizer
        nes-runtime)
