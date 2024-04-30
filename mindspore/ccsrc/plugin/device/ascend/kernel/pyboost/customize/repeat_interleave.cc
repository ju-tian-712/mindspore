/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/ascend/kernel/pyboost/customize/repeat_interleave.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/copy.h"
#include "kernel/pyboost/auto_generate/view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr RepeatInterleaveAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const BaseTensorPtr &input_tensor, const BaseTensorPtr &repeats,
                                                      const std::optional<Int64ImmPtr> &axis,
                                                      const std::optional<Int64ImmPtr> &output_size) {
  OpRunner::InferOpOutput(op, input_tensor, repeats, axis, output_size);

  int64_t axis_imm = axis ? GetValue<int64_t>(*axis) : 0;
  int64_t output_size_imm = output_size ? GetValue<int64_t>(*output_size) : 0;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, repeats);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, repeats, axis_imm, output_size_imm]() {
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         "RepeatInterleave", false);
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, repeats);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnRepeatInterleaveWithDim, device_context, op->stream_id(), input_tensor, repeats, axis_imm,
                   output_size_imm, outputs[0]);
    }));
  return op->outputs()[0];
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
