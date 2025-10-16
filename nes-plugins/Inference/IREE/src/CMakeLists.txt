# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_plugin_as_library(InferenceModel RewriteRule nes-query-optimizer iree_inference_plugin_library
        LowerToPhysicalIreeInference.cpp

        BatchingPhysicalOperator.cpp
        IREEBatchInferenceOperator.cpp
        IREEBatchCacheInferenceOperator.cpp
        IREEBatchInferenceOperatorHandler.cpp

        IREEInferenceOperator.cpp
        IREECacheInferenceOperator.cpp
        IREEInferenceOperatorHandler.cpp

        IREEAdapter.cpp
        IREERuntimeWrapper.cpp

        PredictionCache/PredictionCache.cpp
        PredictionCache/PredictionCache2Q.cpp
        PredictionCache/PredictionCacheFIFO.cpp
        PredictionCache/PredictionCacheLFU.cpp
        PredictionCache/PredictionCacheLRU.cpp
        PredictionCache/PredictionCacheSecondChance.cpp
        PredictionCache/PredictionCacheUtil.cpp

        PredictionCacheBatch/PredictionCacheBatch.cpp
        PredictionCacheBatch/PredictionCacheBatch2Q.cpp
        PredictionCacheBatch/PredictionCacheBatchFIFO.cpp
        PredictionCacheBatch/PredictionCacheBatchLFU.cpp
        PredictionCacheBatch/PredictionCacheBatchLRU.cpp
        PredictionCacheBatch/PredictionCacheBatchSecondChance.cpp
        PredictionCacheBatch/PredictionCacheBatchUtil.cpp
)
find_package(IREERuntime CONFIG REQUIRED)
target_include_directories(iree_inference_plugin_library PRIVATE ../include ../include/PredictionCache)
target_include_directories(iree_inference_plugin_library SYSTEM PRIVATE ${_IREE_COMPILER_API})

target_link_libraries(iree_inference_plugin_library PRIVATE
        iree_runtime_runtime
        nes-physical-operators
        nes-query-optimizer
        nes-runtime
)

add_subdirectory(PredictionCache)
