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
#include "plugin/device/ascend/kernel/opapi/aclnn/anti_quant_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void AntiQuantAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto sqrt_mode = transform::ConvertKernelTensor<bool>(inputs[kIndex3]);
  auto dtype = transform::ConvertKernelTensor<TypeId>(inputs[kIndex4]);
  if (dtype == kNumberTypeBFloat16) {
    dst_type_ = ACL_BF16;
  } else if (dtype == kNumberTypeFloat16) {
    dst_type_ = ACL_FLOAT16;
  } else {
    MS_LOG(EXCEPTION) << "For AntiQuant, 'dtype' only support float16 and bfloat16, but got " << TypeIdToString(dtype);
  }

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], dst_type_, sqrt_mode, outputs[kIndex0]);
}

bool AntiQuantAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto sqrt_mode = transform::ConvertKernelTensor<bool>(inputs[kIndex3]);

  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], dst_type_,
                                      sqrt_mode, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AntiQuant, AntiQuantAscend);
}  // namespace kernel
}  // namespace mindspore
