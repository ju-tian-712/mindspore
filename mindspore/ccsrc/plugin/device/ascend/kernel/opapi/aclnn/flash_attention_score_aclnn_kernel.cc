/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/flash_attention_score_aclnn_kernel.h"
#include <algorithm>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
void FAScoreAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto return_value = FAGenerate(inputs, outputs);
  UpdateWorkspace(return_value);
}

bool FAScoreAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(FAGenerate(inputs, outputs));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLLNN_KERNEL_FACTORY_REG(FlashAttentionScore, FAScoreAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
