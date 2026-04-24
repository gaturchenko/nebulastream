/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <OpenVINOAdapter.hpp>
#include <OpenVINOInferenceOperatorHandler.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

OpenVINOInferenceOperatorHandler::OpenVINOInferenceOperatorHandler(Nebuli::Inference::Model model) : model(std::move(model))
{
}

void OpenVINOInferenceOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    threadLocalAdapters.reserve(pipelineExecutionContext.getNumberOfWorkerThreads());
    for (size_t threadId = 0; threadId < pipelineExecutionContext.getNumberOfWorkerThreads(); ++threadId)
    {
        threadLocalAdapters.emplace_back(OpenVINOAdapter::create());
        threadLocalAdapters.back()->initializeModel(model, 1);
    }
}

void OpenVINOInferenceOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext&)
{
    threadLocalAdapters.clear();
}

const Nebuli::Inference::Model& OpenVINOInferenceOperatorHandler::getModel() const
{
    return model;
}

const std::shared_ptr<OpenVINOAdapter>& OpenVINOInferenceOperatorHandler::getOpenVINOAdapter(WorkerThreadId threadId) const
{
    return threadLocalAdapters[threadId % threadLocalAdapters.size()];
}

}
